#!/usr/bin/env python3
"""GPU fixed-token codec comparison; run only after timed TTS has stopped.

Encode the three retained EN/ES/AR reference WAVs once, then feed exactly the same
codes through serial and grouped codec decoding. This isolates codec behavior;
the inputs are waveform round trips, not the original LLM-generated token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from run import collect_hardware, percentile, write_json
from transformers import HiggsAudioV2TokenizerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--references", type=Path, default=Path(__file__).parent / "inputs/references"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tokens").mkdir()
    for arm in ("serial", "grouped"):
        (args.output / arm / "audio").mkdir(parents=True)
    source = args.source / "src/omnivoice_triton/models/base_runner.py"
    spec = importlib.util.spec_from_file_location("candidate_base_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    hardware = collect_hardware()
    write_json(
        args.output / "metadata.json",
        {
            "hardware": hardware,
            "source_ref": args.source_ref,
            "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model": str(args.model),
            "model_revision": args.model_revision,
            "workload": (
                f"three original EN/ES/AR reference speech WAVs, each repeated "
                f"{args.copies} times; fixed waveform-roundtrip tokens decoded by "
                f"serial and opt-in grouped codec"
            ),
            "co_resident_load": (
                "Must run after TTS timing stops; see nvidia-smi process evidence"
            ),
            "numerical_screen": (
                "max absolute error <=1e-4 and relative RMS error <=1e-4; this screen "
                "is not speech quality certification"
            ),
        },
    )
    selected = json.loads((args.references / "manifest.json").read_text())
    assert len(selected) == 3, "Expected three original EN/ES/AR reference recordings"
    tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        str(args.model / "audio_tokenizer"), device_map="cuda"
    )
    tokenizer.eval()
    results, fixtures = [], []
    with torch.inference_mode():
        for entry in selected:
            path = (args.references / entry["file"]).resolve()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            assert sample_rate == 24000 and waveform.shape[1] == 1
            mono = waveform[:, 0]
            remainder = len(mono) % tokenizer.config.hop_length
            if remainder:
                mono = mono[:-remainder]
            codes = tokenizer.encode(
                torch.from_numpy(mono).view(1, 1, -1).to("cuda")
            ).audio_codes[0]
            case_id = entry["lang"] + "-reference"
            token_path = args.output / "tokens" / (case_id + ".npy")
            np.save(token_path, codes.cpu().numpy())
            fixtures.append(
                {
                    "case_id": case_id,
                    "source": entry,
                    "tokens": str(token_path.relative_to(args.output)),
                    "token_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
                    "shape": list(codes.shape),
                }
            )
            results.extend([codes.clone() for _ in range(args.copies)])

        def decode(batch_size):
            return runner._decode_audio_token_batch(
                results,
                audio_tokenizer=tokenizer,
                sampling_rate=24000,
                cross_fade_chunks=lambda chunks, sr: None,
                max_batch_size=batch_size,
            )

        # Warm both distinct codec batch shapes before the timed comparisons.
        decode(1)
        decode(args.batch_size)
        timings = {"serial": [], "grouped": []}
        outputs = {}
        for repetition in range(args.repetitions):
            order = [("serial", 1), ("grouped", args.batch_size)]
            if repetition % 2:
                order.reverse()
            for arm, size in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                started = time.perf_counter()
                event_start.record()
                audio = decode(size)
                event_end.record()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                timings[arm].append(
                    {
                        "repetition": repetition,
                        "ms": elapsed,
                        "cuda_event_span_ms": event_start.elapsed_time(event_end),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
                outputs[arm] = audio

    comparisons, manifests = [], {"serial": [], "grouped": []}
    for index, fixture in enumerate(fixtures):
        for duplicate in range(args.copies):
            offset = index * args.copies + duplicate
            reference, candidate = outputs["serial"][offset], outputs["grouped"][offset]
            assert reference.shape == candidate.shape
            error = candidate.astype(np.float64) - reference.astype(np.float64)
            rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
            error_rms = float(np.sqrt(np.mean(error**2)))
            maximum = float(np.max(np.abs(error)))
            relative = error_rms / rms if rms else (0.0 if error_rms == 0 else None)
            identity = f"{fixture['case_id']}-copy{duplicate}"
            finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
            snr_db = (
                float(20 * np.log10(rms / error_rms)) if rms and error_rms else None
            )
            comparisons.append(
                {
                    "id": identity,
                    "shape": list(reference.shape),
                    "finite": finite,
                    "bitwise_equal": bool(np.array_equal(reference, candidate)),
                    "max_absolute_error": maximum,
                    "error_rms": error_rms,
                    "reference_rms": rms,
                    "relative_rms_error": relative,
                    "snr_db": snr_db,
                    "snr_infinite": error_rms == 0 and rms > 0,
                    "within_numerical_screen": finite
                    and maximum <= 1e-4
                    and relative is not None
                    and relative <= 1e-4,
                }
            )
            for arm in ("serial", "grouped"):
                path = args.output / arm / "audio" / (identity + ".wav")
                sf.write(path, outputs[arm][offset].squeeze(), 24000, subtype="FLOAT")
                original = fixture["source"]
                language = original["lang"]
                omnilingual_language = {
                    "en": "eng_Latn",
                    "es": "spa_Latn",
                    "ar": "arb_Arab",
                }[language]
                manifests[arm].append(
                    {
                        "id": identity,
                        "arm": "codec-" + arm,
                        "path": "audio/" + path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "language": language,
                        "omnilingual_language": omnilingual_language,
                        "expected": original["text"],
                        "case_id": fixture["case_id"],
                        "phase": "fixed-token-codec-probe",
                        "source": fixture,
                    }
                )
    write_json(args.output / "fixtures.json", fixtures)
    write_json(args.output / "comparisons.json", comparisons)
    write_json(args.output / "timings.json", timings)
    for arm, entries in manifests.items():
        write_json(args.output / arm / "quality-manifest.json", entries)
    result = {
        "gpu": hardware["torch_cuda_devices"],
        "pairs": len(comparisons),
        "all_within_numerical_screen": all(
            row["within_numerical_screen"] for row in comparisons
        ),
        "worst_absolute_error": max(row["max_absolute_error"] for row in comparisons),
        "serial_p50_ms": percentile([row["ms"] for row in timings["serial"]], 50),
        "grouped_p50_ms": percentile([row["ms"] for row in timings["grouped"]], 50),
        "quality_status": (
            "Blind ASR comparison required; numerical screen alone is not qualification"
        ),
    }
    write_json(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
