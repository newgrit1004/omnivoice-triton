#!/usr/bin/env python3
"""Audit retained protocol, actual launch settings and WAVs; no inference.

Reject a mislabeled codec profile or an unapplied requested startup seed even if
all HTTP requests succeeded. Historical runs without a recorded phase selection
use this task's named scenario definitions or disclose limited subset validation.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def flag_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def verify_runtime(root, metadata):
    health = read_json(root / "health-before.json")
    command = metadata["command"]
    checks = {}
    scalar_flags = {
        "--model": ("model", str),
        "--runner": ("runner", str),
        "--device": ("device", str),
        "--dtype": ("dtype", str),
        "--batch-collect-ms": ("batch_collect_ms", float),
        "--max-batch-requests": ("max_batch_requests", int),
        "--max-batch-target-tokens": ("max_batch_target_tokens", int),
        "--max-batch-conditioning-tokens": ("max_batch_conditioning_tokens", int),
        "--max-batch-padding-ratio": ("max_batch_padding_ratio", float),
        "--max-cuda-graphs": ("max_cuda_graphs", int),
        "--decode-postprocess-workers": ("decode_postprocess_workers", int),
        "--audio-decode-batch-size": ("audio_decode_batch_size", int),
    }
    for flag, (key, convert) in scalar_flags.items():
        value = flag_value(command, flag)
        if value is None:
            continue
        assert health.get(key) == convert(value), (
            f"Requested {flag}={value} differs from actual health "
            f"{key}={health.get(key)}"
        )
        checks[flag] = health[key]
    for flag, key in (
        ("--max-pending-requests", "max_pending_requests"),
        ("--max-pending-gpu-tasks", "max_pending_gpu_tasks"),
    ):
        value = flag_value(command, flag)
        if value is not None:
            actual = health["batcher_queue"][key]
            assert actual == int(value), (
                f"Requested {flag} differs from actual queue limit"
            )
            checks[flag] = actual
    buckets = flag_value(command, "--batch-bucket-sizes")
    if buckets:
        assert health["batch_bucket_sizes"] == [int(x) for x in buckets.split(",")]
        checks["--batch-bucket-sizes"] = health["batch_bucket_sizes"]
    for flag, key in (
        ("--sage-attention", "sage_attention"),
        ("--full-triton-patch", "full_triton_patch"),
    ):
        assert health[key] == (flag in command), f"Requested {flag} differs from health"
        checks[flag] = health[key]
    if "--no-asr" in command:
        assert health["load_asr"] is False, "TTS benchmark unexpectedly loaded ASR"
        checks["--no-asr"] = True
    # The original bug omitted the option from the child command itself. The
    # retained scenario name carries the declared intent for that historical arm.
    match = re.search(r"grouped-decode(\d+)", metadata["arm"])
    if match:
        intended = int(match.group(1))
        assert health.get("audio_decode_batch_size") == intended, (
            f"Protocol invalid: {metadata['arm']} intended codec batch {intended}, "
            f"but actual server used {health.get('audio_decode_batch_size')}"
        )
        assert flag_value(command, "--audio-decode-batch-size") == str(intended), (
            "Codec option was not forwarded to server"
        )
    seed = metadata.get("seed")
    requested_seed = seed.get("startup_seed") if isinstance(seed, dict) else None
    if "seeded-critical" in metadata["arm"]:
        assert requested_seed is not None, (
            "Seeded scenario has no requested startup seed"
        )
    if requested_seed is not None:
        evidence_path = root / "server-seed.json"
        assert evidence_path.exists(), (
            "Protocol invalid: requested startup seed has no server application "
            "evidence"
        )
        evidence = read_json(evidence_path)
        assert any(Path(part).name == "seeded_server.py" for part in command), (
            "Startup seed wrapper was not launched"
        )
        assert evidence["requested_seed"] == requested_seed
        assert evidence["torch_initial_seed"] == requested_seed
        assert evidence["cuda_initial_seed"] == requested_seed
        assert evidence["applied_before_module_import"] is True
        assert seed.get("applied") is True, (
            "Startup seed was not verified by the harness"
        )
        checks["startup_seed"] = evidence
    return checks


def expected_phases(metadata, corpus, summaries):
    counts = {
        "smoke": 1,
        "warmup-single": sum(row["length"] == "short" for row in corpus),
        "warmup-burst10": 10,
        "single-corpus": len(corpus),
        "single-short-repeat10": 10,
        "burst10-short": 10,
        "burst10-mixed": len(corpus) + 1,
        "burst100-short": 100,
        "staggered-mixed-0.5rps": len(corpus) * 2,
    }
    for phase in summaries:
        match = re.fullmatch(r"burst(\d+)-overload", phase)
        if match:
            counts[phase] = int(match.group(1))
        assert phase in counts, f"Unknown phase without declared count: {phase}"
    selected = metadata.get("requested_phases", metadata.get("phases"))
    origin = "recorded requested phase selection"
    if isinstance(selected, str):
        selected = [x for x in selected.split(",") if x]
    if not selected:
        arm = metadata["arm"]
        origin = (
            "task scenario definition; historical metadata has no explicit requested "
            "phase list"
        )
        if "seeded-critical" in arm:
            selected = ["warmup-single", "single-corpus"]
        elif arm.endswith("-queue128"):
            selected = ["warmup-single", "warmup-burst10", "burst100-short"]
        elif arm.endswith("-overload"):
            selected = [
                "warmup-single",
                "warmup-burst10",
                "burst1000-overload",
                "burst5000-overload",
            ]
        elif arm in ("before", "async-only", "cache32") or "grouped-decode" in arm:
            selected = [
                key
                for key in counts
                if key != "smoke" and not key.endswith("-overload")
            ]
        elif "smoke" in summaries:
            selected = ["smoke"]
        else:
            selected = list(summaries)
            origin = "observed subset only; requested phase selection unverified"
    for phase in selected:
        if phase not in counts:
            match = re.fullmatch(r"burst(\d+)-overload", phase)
            assert match, f"Unknown requested phase {phase}"
            counts[phase] = int(match.group(1))
    return {phase: counts[phase] for phase in selected}, origin


def verify(root: Path) -> dict:
    root = root.resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    outcome = json.loads((root / "outcome.json").read_text())
    rows = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((root / "quality-manifest.json").read_text())
    summaries = json.loads((root / "summary.json").read_text())
    corpus = read_json(root / "corpus.json")
    runtime_checks = verify_runtime(root, metadata)
    assert len({row["id"] for row in rows}) == len(rows), "Duplicate request IDs"
    assert len({row["id"] for row in manifest}) == len(manifest), (
        "Duplicate quality IDs"
    )
    successes = {row["id"]: row for row in rows if row.get("status") == 200}
    assert set(successes) == {row["id"] for row in manifest}, (
        "ASR manifest does not cover every HTTP 200"
    )
    paths = set()
    for entry in manifest:
        path = (root / entry["path"]).resolve()
        assert path.is_relative_to(root), "Audio escapes retained run directory"
        assert path not in paths, "Audio filename reused across requests"
        paths.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"] == successes[entry["id"]]["sha256"], (
            f"Audio hash mismatch: {entry['id']}"
        )
    for phase, summary in summaries.items():
        phase_rows = [row for row in rows if row["phase"] == phase]
        assert len(phase_rows) == summary["attempts"], f"Phase count mismatch: {phase}"
        assert (
            dict(Counter(str(row.get("status")) for row in phase_rows))
            == summary["status_counts"]
        )
        assert summary["http200"] == sum(row.get("status") == 200 for row in phase_rows)
    if outcome["status"] == "completed":
        expected, phase_evidence = expected_phases(metadata, corpus, summaries)
        assert dict(Counter(row["phase"] for row in rows)) == expected, (
            "Completed run lacks expected requests"
        )
    else:
        raise AssertionError(f"Run is not complete: {outcome['status']}")
    return {
        "root": str(root),
        "arm": metadata["arm"],
        "hardware": metadata["hardware"].get("torch_cuda_devices", []),
        "status": outcome["status"],
        "attempts": len(rows),
        "http200_audio_hashes_verified": len(paths),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "requested_phase_counts": expected,
        "phase_selection_evidence": phase_evidence,
        "runtime_flags_checked": runtime_checks,
        "validation": (
            "CPU-only protocol/retention/accounting check; no inference or quality "
            "certification"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.runs:
        print(json.dumps(verify(directory)))
