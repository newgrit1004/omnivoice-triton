#!/usr/bin/env python3
"""Discover retained WAV manifests, transcribe once per judge, split by arm.

Only start GPU work after all TTS timing/probes are complete. --prepare-only is
CPU/file I/O and may be used to inspect coverage before releasing the GPU.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("omni_evaluate", HERE / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ARM_WARNINGS = {
    "before-seeded-critical": (
        "Harness defect: requested startup seed was recorded in metadata but NOT "
        "applied. This is an unseeded sample set; no seeded pairing claim."
    ),
    "cache32-seeded-critical": (
        "Harness defect: requested startup seed was recorded in metadata but NOT "
        "applied. This is an unseeded sample set; no seeded pairing claim."
    ),
    "grouped-decode8": (
        "Harness defect: main HTTP profile actually used serial codec decode "
        "(health1), despite historical label. This arm does NOT validate grouped codec "
        "decode. Separate fixed-token codec-probe/grouped remains valid."
    ),
}


def prepare_all(root, output, include_arms=None):
    rows, coverage = [], {}
    for manifest in sorted((root / "results").rglob("quality-manifest.json")):
        arm = manifest.parent.relative_to(root / "results").as_posix()
        if "scoring-probe" in arm or "token-probe" in arm:
            continue
        if include_arms and arm not in include_arms:
            continue
        cases = ev.read_rows(manifest)
        ids = set()
        for case in cases:
            case = dict(case)
            original_id = str(case["id"])
            if original_id in ids:
                raise ValueError(f"duplicate id in {arm}: {original_id}")
            ids.add(original_id)
            wav = Path(case["path"])
            if not wav.is_absolute():
                wav = manifest.parent / wav
            case.update(
                {
                    "id": f"{arm}::{original_id}",
                    "original_id": original_id,
                    "quality_arm": arm,
                    "run": arm,
                    "path": str(wav.resolve()),
                    "quality_manifest": str(manifest),
                }
            )
            rows.append(case)
        request_log = manifest.parent / "requests.jsonl"
        item = {
            "manifest": str(manifest),
            "manifest_sha256": ev.sha256(manifest),
            "retained_audio_cases": len(cases),
            "retained_audio_ids": sorted(ids),
            "unique_original_audio_hashes": len({x["sha256"] for x in cases}),
        }
        if arm in ARM_WARNINGS:
            item["provenance_warning"] = ARM_WARNINGS[arm]
        if request_log.exists():
            requests = ev.read_rows(request_log)
            statuses = Counter(
                str(x.get("status", "transport_error")) for x in requests
            )
            successful = {str(x["id"]) for x in requests if x.get("status") == 200}
            item.update(
                {
                    "request_log": str(request_log),
                    "request_log_sha256": ev.sha256(request_log),
                    "attempts": len(requests),
                    "http_status_counts": dict(statuses),
                    "http200_count": statuses.get("200", 0),
                    "http200_without_retained_audio": sorted(successful - ids),
                    "retained_without_http200_record": sorted(ids - successful),
                    "rejected_or_failed_requests": [
                        {
                            "id": x.get("id"),
                            "phase": x.get("phase"),
                            "status": x.get("status"),
                            "error": x.get("error"),
                        }
                        for x in requests
                        if x.get("status") != 200
                    ],
                }
            )
        else:
            item["note"] = (
                "Codec/control fixture; no HTTP request availability denominator."
            )
        coverage[arm] = item
    if include_arms and set(include_arms) - coverage.keys():
        raise ValueError(
            f"Requested arms not found: {sorted(set(include_arms) - coverage.keys())}"
        )
    controls = HERE / "control-manifest.json"
    control_rows = ev.read_rows(controls)
    for case in control_rows:
        case = dict(case)
        original_id = case["id"]
        wav = Path(case["path"])
        if not wav.is_absolute():
            wav = controls.parent / wav
        case.update(
            {
                "id": f"reference-controls::{original_id}",
                "original_id": original_id,
                "quality_arm": "reference-controls",
                "run": "reference-controls",
                "path": str(wav.resolve()),
                "quality_manifest": str(controls),
            }
        )
        rows.append(case)
    coverage["reference-controls"] = {
        "retained_audio_cases": len(control_rows),
        "manifest": str(controls),
        "manifest_sha256": ev.sha256(controls),
        "note": (
            "Unchanged original reference recordings, calibration only; no HTTP "
            "denominator."
        ),
    }
    if not rows:
        raise ValueError("No retained audio discovered")
    # Calibrate each judge on unchanged known recordings before synthesized WAVs.
    rows.sort(
        key=lambda x: (
            0 if x["quality_arm"] == "reference-controls" else 1,
            x["quality_arm"],
            x["id"],
        )
    )
    global_ids = [x["id"] for x in rows]
    if len(set(global_ids)) != len(rows):
        raise ValueError("Duplicate combined IDs")
    ev.write_json(output / "combined-manifest.json", rows)
    ev.write_json(
        output / "http-and-audio-coverage.json",
        {
            "measurement_kind": "File/HTTP coverage bookkeeping, no GPU inference",
            "total_retained_audio_cases": len(rows),
            "unique_original_audio_hashes": len({x["sha256"] for x in rows}),
            "arms": coverage,
            "note": (
                "ASR only screens saved successful WAVs. Rejections/errors remain in "
                "HTTP denominators; no quality pass is inferred for missing audio. "
                "Token-only scoring probes need no ASR and are excluded."
            ),
        },
    )
    print(
        json.dumps(
            {
                "retained_audio_cases": len(rows),
                "unique_audio_hashes": len({x["sha256"] for x in rows}),
                "arms": {k: v["retained_audio_cases"] for k, v in coverage.items()},
            },
            indent=2,
        ),
        flush=True,
    )


def split_results(output, models):
    combined = output / "combined"
    prepared = ev.read_rows(combined / "prepared.json")
    transcripts = {
        model: {x["id"]: x for x in ev.read_rows(combined / f"{model}.jsonl")}
        for model in models
    }
    arms = sorted({x["source"]["quality_arm"] for x in prepared})
    index = {}
    for arm in arms:
        directory = output / "by-arm" / arm
        directory.mkdir(parents=True, exist_ok=True)
        selected = [x for x in prepared if x["source"]["quality_arm"] == arm]
        arm_prepared = []
        for row in selected:
            arm_prepared.append(
                {**row, "id": row["source"]["original_id"], "combined_id": row["id"]}
            )
        ev.write_json(directory / "prepared.json", arm_prepared)
        for model in models:
            with (directory / f"{model}.jsonl").open("w") as f:
                for row in selected:
                    transcript = transcripts[model][row["id"]]
                    f.write(
                        json.dumps(
                            {
                                **transcript,
                                "id": row["source"]["original_id"],
                                "combined_id": row["id"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            shutil.copy2(
                combined / f"{model}-metadata.json",
                directory / f"{model}-metadata.json",
            )
        ev.score(directory, models)
        index[arm] = json.loads((directory / "summary.json").read_text())
        if arm in ARM_WARNINGS:
            index[arm]["provenance_warning"] = ARM_WARNINGS[arm]
            ev.write_json(directory / "summary.json", index[arm])
    pairs = [
        ("before", "async-only"),
        ("before-queue128", "async-queue128"),
        ("before", "grouped-decode8"),
        ("codec-probe/serial", "codec-probe/grouped"),
        ("before-overload", "safe-overload"),
        ("async-overload", "safe-overload"),
        ("before", "cache32"),
        ("async-only", "cache32"),
    ]
    pairs += [
        ("before-overload", arm) for arm in arms if arm.startswith("async-overload")
    ]
    if "async-overload" in arms:
        pairs += [
            ("async-overload", arm)
            for arm in arms
            if arm.startswith("async-overload") and arm != "async-overload"
        ]
    # Automatically pair additional nested startup-seeded before/candidate arms.
    for arm in arms:
        if "seed" in arm and "before" in arm:
            candidates = [
                x for x in arms if "seed" in x and "before" not in x and x != arm
            ]
            pairs.extend((arm, candidate) for candidate in candidates)
    pairs = list(dict.fromkeys(pairs))
    comparisons = {}
    for baseline, candidate in pairs:
        if baseline not in arms or candidate not in arms:
            continue
        name = baseline.replace("/", "-") + "-vs-" + candidate.replace("/", "-")
        destination = output / "comparisons" / name
        subprocess.run(
            [
                sys.executable,
                str(HERE / "compare.py"),
                "--baseline",
                str(output / "by-arm" / baseline / "scored.json"),
                "--candidate",
                str(output / "by-arm" / candidate / "scored.json"),
                "--output",
                str(destination),
            ],
            check=True,
        )
        comparisons[name] = json.loads((destination / "comparison.json").read_text())[
            "summary"
        ]
    ev.write_json(
        output / "index.json",
        {
            "arms": index,
            "comparisons": comparisons,
            "scope": (
                "Offline GPU ASR screens of retained audio; human/perceptual review "
                "remains necessary. Original HTTP availability coverage is separate."
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/omnivoice-speedups")
    parser.add_argument("--output")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--models", nargs="+", default=ev.DEFAULT_MODELS)
    parser.add_argument(
        "--include-arms",
        nargs="+",
        help=(
            "Exact relative result-arm paths for a small follow-up; reference controls "
            "are always added"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else root / "asr-results"
    output.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        prepare_all(root, output, args.include_arms)
        if args.prepare_only:
            return
        subprocess.run(
            [
                sys.executable,
                str(HERE / "evaluate.py"),
                "--manifest",
                str(output / "combined-manifest.json"),
                "--output",
                str(output / "combined"),
                "--models",
                *args.models,
            ],
            check=True,
        )
    split_results(output, args.models)


if __name__ == "__main__":
    main()
