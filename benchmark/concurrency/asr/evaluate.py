#!/usr/bin/env python3
"""Blind, local GPU Omnilingual judges; scoring never feeds reference text to ASR.

Run after TTS timing, never concurrently with timed TTS. Each model is loaded in
its own subprocess, so model VRAM is released before the second model starts.
Input: JSON/JSONL records with id, path, text (or expected), and language.
Supported shorthand en/es/ar maps to Meta language+script IDs. Other languages
must provide lang_id, verified against the installed supported_langs list.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

DEFAULT_MODELS = ["omniASR_LLM_Unlimited_1B_v2", "omniASR_CTC_1B_v2"]
LANGS = {"en": "eng_Latn", "es": "spa_Latn", "ar": "arb_Arab"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    raw = Path(path).read_text()
    return (
        json.loads(raw)
        if raw.lstrip().startswith("[")
        else [json.loads(x) for x in raw.splitlines() if x.strip()]
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(text):
    # Broad NFKC/case/punctuation normalization. Keep lexical spelling differences.
    # Arabic-specific alef/ya normalization is ONLY applied to Arabic characters.
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(
        " " if unicodedata.category(c).startswith(("P", "S")) else c for c in value
    )
    value = "".join(
        c
        for c in value
        if c != "\u0640" and not "\u064b" <= c <= "\u065f" and c != "\u0670"
    )
    value = value.translate(
        str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
    )
    return " ".join(value.split())


def hardware():
    import torch

    packages = {}
    for p in [
        "omnilingual-asr",
        "fairseq2",
        "fairseq2n",
        "torch",
        "torchaudio",
        "numpy",
        "soundfile",
        "jiwer",
    ]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None

    def query(fields):
        try:
            return subprocess.check_output(["nvidia-smi", *fields], text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packages": packages,
        "python": sys.version,
        "cuda_runtime": torch.version.cuda,
        "physical_gpu_nvidia_smi": query(
            [
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "co_resident_gpu_processes": query(
            [
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        ),
        "torch_visible_gpu": [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        "measurement_kind": "GPU ASR inference, separate from TTS latency measurements",
    }


def prepare(manifest, output):
    import numpy as np
    import soundfile as sf

    rows = read_rows(manifest)
    seen = set()
    prepared = []
    chunk_dir = output / "asr-input"
    chunk_dir.mkdir(exist_ok=True)
    for n, source in enumerate(rows):
        request = source.get("request", source)
        case_id = str(source.get("id", source.get("request_id", n)))
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id}")
        seen.add(case_id)
        path = Path(source["path"])
        if not path.is_absolute():
            path = Path(manifest).parent / path
        path = path.resolve()
        digest = sha256(path)
        if source.get("sha256") and source["sha256"] != digest:
            raise ValueError(f"audio hash mismatch {case_id}")
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"empty/nonfinite audio {case_id}")
        lang = source.get(
            "lang_id",
            source.get("omnilingual_language", LANGS.get(source.get("language", "en"))),
        )
        if not lang:
            raise ValueError(f"set explicit Meta lang_id for case {case_id}")
        chunks = []
        # Keep inputs <=30s for both judges. Search 22-29s for a quiet boundary.
        # ASR-only chunks never replace or alter the complete original WAV.
        offset = 0
        while offset < len(audio):
            if len(audio) - offset <= 30 * sr:
                end = len(audio)
            else:
                frame = max(1, sr // 50)
                candidates = range(offset + 22 * sr, offset + 29 * sr, frame)
                end = min(
                    candidates, key=lambda x: float(np.mean(audio[x : x + frame] ** 2))
                )
            chunk_path = chunk_dir / f"{n:06d}-{len(chunks):03d}.wav"
            sf.write(chunk_path, audio[offset:end], sr, subtype="FLOAT")
            chunks.append(
                {
                    "path": str(chunk_path.resolve()),
                    "start_s": offset / sr,
                    "end_s": end / sr,
                    "sha256": sha256(chunk_path),
                }
            )
            offset = end
        prepared.append(
            {
                "id": case_id,
                "path": str(path),
                "sha256": digest,
                "text": source.get("expected", request.get("text")),
                "language": source.get("language"),
                "lang_id": lang,
                "chunks": chunks,
                "duration_s": len(audio) / sr,
                "audio": {
                    "sample_rate": sr,
                    "peak": float(np.max(np.abs(audio))),
                    "rms": float(np.sqrt(np.mean(audio**2))),
                    "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 0.999)),
                },
                "run": source.get("run"),
                "phase": source.get("phase", request.get("phase")),
                "source": source,
            }
        )
        if not isinstance(prepared[-1]["text"], str):
            raise ValueError(f"missing expected text for {case_id}")
    write_json(output / "prepared.json", prepared)


def run_model(args):
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")
    cases = read_rows(Path(args.output) / "prepared.json")
    for case in cases:
        if case["lang_id"] not in supported_langs:
            raise ValueError(f"unsupported lang_id {case['lang_id']}")
    meta = hardware()
    meta.update(
        {
            "model_card": args.worker_model,
            "dtype": args.dtype,
            "device": args.device,
            "confidence": None,
            "confidence_note": (
                "Reference pipeline returns text only; no calibrated confidence "
                "available."
            ),
            "blind": True,
            "batch_size": 1,
            "language_conditioning": "known language/script only; ignored by CTC",
            "expected_text_prompt": False,
            "supported_language_script_ids": len(supported_langs),
        }
    )
    dtype = getattr(torch, args.dtype)
    started = time.perf_counter()
    pipeline = ASRInferencePipeline(
        model_card=args.worker_model, device=args.device, dtype=dtype
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    meta["load_wall_s"] = time.perf_counter() - started
    write_json(Path(args.output) / f"{args.worker_model}-metadata.json", meta)
    out = Path(args.output) / f"{args.worker_model}.jsonl"
    transcripts_by_audio = {}
    with out.open("w") as f:
        for case in cases:
            cache_key = (case["sha256"], case["lang_id"])
            if cache_key in transcripts_by_audio:
                original = transcripts_by_audio[cache_key]
                reused = {
                    **original,
                    "id": case["id"],
                    "asr_wall_s": 0.0,
                    "peak_allocated_bytes": None,
                    "asr_reused_from_id": original["id"],
                }
                f.write(json.dumps(reused, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    json.dumps(
                        {
                            "id": case["id"],
                            "model": args.worker_model,
                            "asr_reused_from_id": original["id"],
                        }
                    ),
                    flush=True,
                )
                continue
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            # Only paths and language codes go to transcribe; reference excluded.
            texts = pipeline.transcribe(
                [x["path"] for x in case["chunks"]],
                lang=[case["lang_id"]] * len(case["chunks"]),
                batch_size=1,
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            row = {
                "id": case["id"],
                "sha256": case["sha256"],
                "model_card": args.worker_model,
                "transcript": " ".join(x.strip() for x in texts),
                "chunk_transcripts": texts,
                "asr_wall_s": time.perf_counter() - start,
                "confidence": None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device.startswith("cuda")
                else None,
            }
            transcripts_by_audio[cache_key] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "model": args.worker_model,
                        "asr_wall_s": row["asr_wall_s"],
                    }
                ),
                flush=True,
            )


def score(output, models):
    import jiwer

    cases = read_rows(output / "prepared.json")
    all_transcripts = {
        model: {x["id"]: x for x in read_rows(output / f"{model}.jsonl")}
        for model in models
    }
    results = []
    for case in cases:
        ref = normalize(case["text"])
        judge_rows = []
        for model in models:
            transcript = all_transcripts[model][case["id"]]
            if transcript["sha256"] != case["sha256"]:
                raise ValueError("transcript/audio mismatch")
            hyp = normalize(transcript["transcript"])
            words = jiwer.process_words(ref, hyp)
            chars = jiwer.process_characters(ref.replace(" ", ""), hyp.replace(" ", ""))
            edits = []
            for a in words.alignments[0]:
                if a.type != "equal":
                    edits.append(
                        {
                            "type": a.type,
                            "expected": ref.split()[a.ref_start_idx : a.ref_end_idx],
                            "recognized": hyp.split()[a.hyp_start_idx : a.hyp_end_idx],
                        }
                    )
            judge_rows.append(
                {
                    **transcript,
                    "normalized_transcript": hyp,
                    "wer": words.wer,
                    "cer": chars.cer,
                    "substitutions": words.substitutions,
                    "deletions": words.deletions,
                    "insertions": words.insertions,
                    "hits": words.hits,
                    "reference_words": len(ref.split()),
                    "reference_characters": len(ref.replace(" ", "")),
                    "character_edits": chars.substitutions
                    + chars.deletions
                    + chars.insertions,
                    "exact_normalized": ref == hyp,
                    "spacing_only_difference": ref != hyp
                    and ref.replace(" ", "") == hyp.replace(" ", ""),
                    "exact_characters_ignoring_whitespace": ref.replace(" ", "")
                    == hyp.replace(" ", ""),
                    "edits": edits,
                }
            )
        normalized = {x["normalized_transcript"] for x in judge_rows}
        flags = []
        if len(normalized) > 1:
            flags.append("asr_disagreement")
        if any(not x["exact_normalized"] for x in judge_rows):
            flags.append("word_mismatch_review")
        if any(x["spacing_only_difference"] for x in judge_rows):
            flags.append("asr_spacing_only_difference_present")
        if len(case["chunks"]) > 1:
            flags.append("asr_chunk_boundaries_review")
        if case["audio"]["clipped_sample_fraction"] > 0:
            flags.append("clipped_samples_review")
        results.append(
            {
                **case,
                "judges": judge_rows,
                "flags": flags,
                "status": "review_required" if flags else "asr_screen_pass",
                "note": (
                    "ASR screen is not perceptual quality certification; "
                    "disagreement/mismatch is not proof of TTS error. WER for "
                    "languages without space-delimited words is not comparable; "
                    "inspect CER."
                ),
            }
        )
    write_json(output / "scored.json", results)
    summary = {
        "cases": len(results),
        "unique_original_audio_hashes": len({x["sha256"] for x in results}),
        "flagged_cases": sum(x["status"] == "review_required" for x in results),
        "status": "review_required"
        if any(x["status"] == "review_required" for x in results)
        else "asr_screen_pass",
        "models": {},
        "languages_observed": sorted({x["lang_id"] for x in results}),
        "limitations": [
            (
                "Two variants share the Omnilingual family and are not independent "
                "ground truth."
            ),
            "No MOS, human listening, or speaker similarity measurement.",
            "No calibrated ASR confidence available.",
            "Language support is not per-language quality qualification.",
        ],
    }
    for model in models:
        rows = [
            next(j for j in x["judges"] if j["model_card"] == model) for x in results
        ]
        edits = sum(x["substitutions"] + x["deletions"] + x["insertions"] for x in rows)
        words = sum(x["reference_words"] for x in rows)
        characters = sum(x["reference_characters"] for x in rows)
        summary["models"][model] = {
            "exact_cases": sum(x["exact_normalized"] for x in rows),
            "word_edits": edits,
            "spacing_only_cases": sum(x["spacing_only_difference"] for x in rows),
            "exact_character_cases_ignoring_whitespace": sum(
                x["exact_characters_ignoring_whitespace"] for x in rows
            ),
            "actual_asr_inferences": sum(not x.get("asr_reused_from_id") for x in rows),
            "reference_words": words,
            "aggregate_wer": edits / words if words else None,
            "aggregate_cer": sum(x["character_edits"] for x in rows) / characters
            if characters
            else None,
            "asr_wall_s": sum(x["asr_wall_s"] for x in rows),
        }
        summary["models"][model]["per_language"] = {}
        for language in summary["languages_observed"]:
            subset = [
                next(j for j in x["judges"] if j["model_card"] == model)
                for x in results
                if x["lang_id"] == language
            ]
            word_count = sum(x["reference_words"] for x in subset)
            char_count = sum(x["reference_characters"] for x in subset)
            word_edits = sum(
                x["substitutions"] + x["deletions"] + x["insertions"] for x in subset
            )
            summary["models"][model]["per_language"][language] = {
                "cases": len(subset),
                "exact_cases": sum(x["exact_normalized"] for x in subset),
                "spacing_only_cases": sum(x["spacing_only_difference"] for x in subset),
                "reference_words": word_count,
                "word_edits": word_edits,
                "substitutions": sum(x["substitutions"] for x in subset),
                "deletions": sum(x["deletions"] for x in subset),
                "insertions": sum(x["insertions"] for x in subset),
                "aggregate_wer": word_edits / word_count if word_count else None,
                "aggregate_cer": sum(x["character_edits"] for x in subset) / char_count
                if char_count
                else None,
            }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    if args.worker_model:
        run_model(args)
        return
    if not args.score_only:
        if not args.manifest:
            parser.error("--manifest is required unless --score-only")
        if any((output / f"{x}.jsonl").exists() for x in args.models):
            raise FileExistsError(
                "Use a fresh output directory to preserve previous transcripts."
            )
        prepare(Path(args.manifest).resolve(), output)
        for model in args.models:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output),
                    "--worker-model",
                    model,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                ],
                check=True,
            )
    score(output, args.models)


if __name__ == "__main__":
    main()
