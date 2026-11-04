#!/usr/bin/env python3
"""Compare original paired WAV hashes/samples without alignment or resampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load(manifest):
    path = Path(manifest)
    rows = json.loads(path.read_text())
    output = {}
    for row in rows:
        wav = Path(row["path"])
        if not wav.is_absolute():
            wav = path.parent / wav
        wav = wav.resolve()
        digest = hashlib.sha256(wav.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Original WAV hash changed: {wav}")
        output[row["id"]] = {**row, "path": str(wav)}
    if len(output) != len(rows):
        raise ValueError("Duplicate IDs")
    return output


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--before", required=True)
parser.add_argument("--after", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
before, after = load(args.before), load(args.after)


def tts_hardware(manifest):
    path = Path(manifest).parent / "metadata.json"
    if not path.exists():
        return {"status": "unverified; source hardware metadata missing"}
    data = json.loads(path.read_text())
    hardware = data.get("hardware", {})
    seed_file = path.parent / "server-seed.json"
    seed_evidence = json.loads(seed_file.read_text()) if seed_file.exists() else None
    return {
        "evidence_file": str(path),
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed_metadata": data.get("seed"),
        "server_seed_evidence": seed_evidence,
        "server_seed_evidence_file": str(seed_file) if seed_file.exists() else None,
        "server_seed_evidence_sha256": hashlib.sha256(
            seed_file.read_bytes()
        ).hexdigest()
        if seed_file.exists()
        else None,
        **{
            k: hardware.get(k)
            for k in [
                "collected_at",
                "nvidia_smi_gpu",
                "torch_cuda_devices",
                "torch_cuda_runtime",
                "packages",
                "nvidia_smi_processes",
            ]
        },
    }


pairs = []
for key in sorted(before.keys() & after.keys()):
    b, a = before[key], after[key]
    if b["expected"] != a["expected"]:
        raise ValueError(f"Expected text differs for pair {key}")
    x, xs = sf.read(b["path"], dtype="float64", always_2d=True)
    y, ys = sf.read(a["path"], dtype="float64", always_2d=True)
    comparable = xs == ys and x.shape == y.shape
    delta = x - y if comparable else None
    pairs.append(
        {
            "id": key,
            "language": b.get("language"),
            "expected": b["expected"],
            "before_path": b["path"],
            "after_path": a["path"],
            "before_sha256": b["sha256"],
            "after_sha256": a["sha256"],
            "identical_wav_sha256": b["sha256"] == a["sha256"],
            "before_sample_rate": xs,
            "after_sample_rate": ys,
            "before_shape": list(x.shape),
            "after_shape": list(y.shape),
            "before_duration_s": len(x) / xs,
            "after_duration_s": len(y) / ys,
            "sample_comparison_valid": comparable,
            "identical_stored_waveform_samples": bool(np.array_equal(x, y))
            if comparable
            else False,
            "max_absolute_sample_difference": float(np.max(np.abs(delta)))
            if comparable
            else None,
            "rms_sample_difference": float(np.sqrt(np.mean(delta**2)))
            if comparable
            else None,
            "mean_absolute_sample_difference": float(np.mean(np.abs(delta)))
            if comparable
            else None,
        }
    )
report = {
    "measurement_kind": (
        "CPU-only comparison of retained TTS WAV files; no new model inference"
    ),
    "source_tts_hardware": {
        "before": tts_hardware(args.before),
        "after": tts_hardware(args.after),
    },
    "tts_hardware_evidence": [
        str(Path(args.before).parent / "metadata.json"),
        str(Path(args.after).parent / "metadata.json"),
    ],
    "pairs": pairs,
    "paired_cases": len(pairs),
    "identical_wav_sha256_pairs": sum(x["identical_wav_sha256"] for x in pairs),
    "identical_stored_waveform_pairs": sum(
        x["identical_stored_waveform_samples"] for x in pairs
    ),
    "same_shape_and_sample_rate_pairs": sum(
        x["sample_comparison_valid"] for x in pairs
    ),
    "missing_before_ids": sorted(after.keys() - before.keys()),
    "missing_after_ids": sorted(before.keys() - after.keys()),
    "note": (
        "No alignment, resampling, amplitude normalization or cropping. Different "
        "shapes/sample rates have no numerical-difference score. Waveform difference "
        "alone is not a perceptual-quality verdict. Actual startup-seed application "
        "requires separate validated server evidence; API exposes no per-request seed."
    ),
}
Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in report.items() if k != "pairs"}, ensure_ascii=False, indent=2
    )
)
