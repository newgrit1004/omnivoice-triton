#!/usr/bin/env python3
"""Finish CPU evidence/provenance, then write the root's ASR retention marker."""

import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/omnivoice-speedups")
RESULTS = ROOT / "asr-results"
NOTE = (
    "Missing IDs can reflect non-200 HTTP outcomes or incomplete ASR coverage; "
    "join request logs and never drop them from availability denominators. "
    "The API has no per-request seed. Verified startup seeding plus matched "
    "sequential order may produce identical WAVs; inspect server-seed evidence and "
    "reported SHA equality. "
    "Unseeded or concurrently scheduled samples remain stochastic. ASR flags "
    "require review and are not proven TTS defects."
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


provenance = RESULTS / "provenance"
provenance.mkdir(exist_ok=True)
for source, destination in [
    (ROOT / "asr-assets.json", provenance / "checkpoint-tokenizer-sha256.json"),
    (ROOT / "asr-card-hashes.json", provenance / "installed-model-card-sha256.json"),
    (ROOT / "asr-venv/installed-packages.txt", provenance / "installed-packages.txt"),
]:
    shutil.copy2(source, destination)

# Correct explanatory wording without changing strict metric values/transcripts.
for path in RESULTS.rglob("comparison.json"):
    data = read(path)
    old_note = data["summary"]["note"]
    data["summary"]["note"] = NOTE
    if path.parent.name == "before-seeded-critical-vs-cache32-seeded-critical":
        data["summary"]["provenance_warning"] = (
            "Requested seed was metadata-only and NOT applied. Historical samples "
            "are unseeded; corrected experiment is separate."
        )
    if path.parent.name == "before-vs-grouped-decode8":
        data["summary"]["provenance_warning"] = (
            "Historical grouped-decode8 HTTP arm actually used serial decode; "
            "corrected actual grouped8 experiment is separate."
        )
    if (
        path.parent.name
        == "before-seeded-critical-corrected-vs-cache32-seeded-critical-corrected"
    ):
        data["summary"]["verified_preservation_evidence"] = (
            "All18 original WAV SHA256 and stored-sample pairs identical under "
            "verified startup seed20261103 and matched sequential order; "
            "per-request seeds are not exposed. Existing baseline flags remain "
            "unchanged."
        )
    write(path, data)
    markdown = path.parent / "review.md"
    if markdown.exists():
        markdown.write_text(markdown.read_text().replace(old_note, NOTE))

for directory in [RESULTS, RESULTS / "corrected"]:
    index_path = directory / "index.json"
    index = read(index_path)
    for name in index.get("comparisons", {}):
        comparison = directory / "comparisons" / name / "comparison.json"
        if comparison.exists():
            index["comparisons"][name] = read(comparison)["summary"]
    write(index_path, index)

passes = {}
all_cases = []
totals = Counter()
model_calls = Counter()
model_segments = Counter()
for name, directory in [("initial", RESULTS), ("corrected", RESULTS / "corrected")]:
    coverage = read(directory / "http-and-audio-coverage.json")
    prepared = read(directory / "combined/prepared.json")
    scored = read(directory / "combined/scored.json")
    summary = read(directory / "combined/summary.json")
    assert len(prepared) == len(scored) == summary["cases"]
    identifiers = {x["id"]: x for x in prepared}
    unique = {}
    for row in prepared:
        unique.setdefault((row["sha256"], row["lang_id"]), row)
    statuses = Counter()
    missing_audio, missing_http = [], []
    for arm, item in coverage["arms"].items():
        statuses.update(item.get("http_status_counts", {}))
        missing_audio.extend(
            {"arm": arm, "id": x}
            for x in item.get("http200_without_retained_audio", [])
        )
        missing_http.extend(
            {"arm": arm, "id": x}
            for x in item.get("retained_without_http200_record", [])
        )
    models = {}
    for model, metrics in summary["models"].items():
        transcriptions = [
            json.loads(x)
            for x in (directory / "combined" / f"{model}.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(transcriptions) == len(prepared)
        assert {x["id"] for x in transcriptions} == identifiers.keys()
        assert all(
            x["sha256"] == identifiers[x["id"]]["sha256"] for x in transcriptions
        )
        actual = [x for x in transcriptions if not x.get("asr_reused_from_id")]
        segments = sum(len(identifiers[x["id"]]["chunks"]) for x in actual)
        assert len(actual) == len(unique) == metrics["actual_asr_inferences"]
        models[model] = {
            "logical_wav_cases_screened": len(transcriptions),
            "actual_unique_wav_transcribe_calls": len(actual),
            "asr_input_segments": segments,
            "peak_torch_allocated_bytes": max(
                x.get("peak_allocated_bytes") or 0 for x in actual
            ),
            **{
                k: metrics.get(k)
                for k in [
                    "exact_cases",
                    "spacing_only_cases",
                    "exact_character_cases_ignoring_whitespace",
                    "aggregate_wer",
                    "aggregate_cer",
                ]
            },
        }
        model_calls[model] += len(actual)
        model_segments[model] += segments
    pass_report = {
        "logical_saved_audio_cases": len(scored),
        "unique_original_wav_hashes": len({x["sha256"] for x in scored}),
        "unique_wav_hash_and_language_keys": len(unique),
        "models": models,
        "http_attempts": sum(statuses.values()),
        "http_status_counts": dict(statuses),
        "http200_without_retained_audio": missing_audio,
        "retained_audio_without_http200": missing_http,
        "review_flagged_cases": summary["flagged_cases"],
        "flag_types": dict(Counter(flag for row in scored for flag in row["flags"])),
        "source_coverage": str(directory / "http-and-audio-coverage.json"),
        "source_scored": str(directory / "combined/scored.json"),
        "clipping_screen_cases": [
            {"id": x["id"], "path": x["path"], "audio": x["audio"]}
            for x in scored
            if "clipped_samples_review" in x["flags"]
        ],
    }
    passes[name] = pass_report
    totals.update(statuses)
    all_cases.extend(prepared)

pair_report = read(RESULTS / "corrected/paired-audio-samples.json")
assert (
    pair_report["paired_cases"]
    == pair_report["identical_wav_sha256_pairs"]
    == pair_report["identical_stored_waveform_pairs"]
    == 18
)
assert all(
    x["max_absolute_sample_difference"] == x["rms_sample_difference"] == 0
    for x in pair_report["pairs"]
)
for item in pair_report["source_tts_hardware"].values():
    seed = item["server_seed_evidence"]
    assert item["seed_metadata"]["applied"] is True
    assert (
        seed["requested_seed"]
        == seed["torch_initial_seed"]
        == seed["cuda_initial_seed"]
        == 20261103
    )
    assert seed["applied_before_module_import"] is True

# Add the final cross-arm grouped comparison to the corrected index.
index_path = RESULTS / "corrected/index.json"
index = read(index_path)
comparison = (
    RESULTS
    / "corrected/comparisons/before-vs-grouped-decode8-corrected/comparison.json"
)
index["comparisons"]["before-vs-grouped-decode8-corrected"] = read(comparison)[
    "summary"
]
index["corrected_startup_seed_sample_evidence"] = str(
    RESULTS / "corrected/paired-audio-samples.json"
)
index["cross_arm_grouped_comparison_note"] = (
    "Unseeded stochastic mixed HTTP profiles; missing pairs reflect different "
    "rejection IDs, so this is not a causal quality comparison."
)
write(index_path, index)

script_rows = [
    {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
    for p in sorted((ROOT / "asr").iterdir())
    if p.is_file() and p.suffix in [".py", ".sh", ".md", ".json"]
]
write(
    provenance / "final-evaluation-scripts-sha256.json",
    {
        "scope": (
            "Final retained evaluation/scoring scripts. During-run edits added CPU "
            "diagnostics and provenance; model, audio, language and inference settings "
            "were unchanged. Initial pass was CPU-rescored with the final spacing "
            "diagnostics."
        ),
        "files": script_rows,
    },
)

global_report = {
    "source_tts_and_asr_gpu": (
        "NVIDIA GeForce RTX3090, physical24576MiB, torch-visible25296044032bytes, "
        "driver580.65.06, CUDA12.8"
    ),
    "date_operator": "2026-11-03 America/Chicago",
    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "passes": passes,
    "total_logical_saved_audio_cases_across_passes": len(all_cases),
    "distinct_original_wav_hashes_across_passes": len({x["sha256"] for x in all_cases}),
    "distinct_original_wav_paths_across_passes": len({x["path"] for x in all_cases}),
    "http_attempts": sum(totals.values()),
    "http_status_counts": dict(totals),
    "actual_unique_wav_transcribe_calls_by_model": dict(model_calls),
    "asr_input_segments_by_model": dict(model_segments),
    "total_review_flagged_logical_cases": sum(
        p["review_flagged_cases"] for p in passes.values()
    ),
    "tested_language_scripts": ["eng_Latn", "spa_Latn", "arb_Arab"],
    "installed_supported_language_script_count": 1672,
    "corrected_seed_preservation": {
        "pairs": 18,
        "warmups": 3,
        "text_cases": 15,
        "identical_wav_sha256": 18,
        "identical_stored_waveforms": 18,
        "maximum_sample_difference": 0,
        "startup_seed": 20261103,
        "server_seed_evidence_verified": True,
        "per_request_seed_available": False,
        "paired_asr_score_changes": 0,
        "existing_review_flags_per_arm": 11,
        "evidence": str(RESULTS / "corrected/paired-audio-samples.json"),
    },
    "interpretation": [
        (
            "All successful retained TTS WAVs were screened. HTTP429 responses are "
            "overload rejections, not successful low-latency capacity."
        ),
        (
            "Review flags are not a TTS failure rate; many are ASR whitespace/number "
            "orthography, decoder disagreement or ASR chunk-boundary exposure."
        ),
        (
            "Each pass repeats the same three reference controls; duplicate codec "
            "samples and identical corrected before/after WAVs are not independent "
            "quality evidence."
        ),
        (
            "CTC transcribed all three original controls exactly; LLM repeated one "
            "final Arabic word. Two Omnilingual variants are not independent ground "
            "truth."
        ),
        (
            "Purposive offline Whisper cross-checks on Apple M5 Pro24GiB (12 clips) "
            "found both cleared ASR artifacts and persistent word/intelligibility "
            "concerns. They are separate from RTX3090 measurements and are not a "
            "population failure-rate sample."
        ),
        (
            "The corrected cache32 sequential fixtures preserve exact audio, including "
            "existing baseline problems. Concurrent unseeded cohorts do not establish "
            "a causal quality win/loss."
        ),
        (
            "No human listening, MOS or speaker-similarity certification was "
            "performed. Three tested languages do not qualify100+ languages."
        ),
    ],
    "provenance": {
        "checkpoints_and_tokenizer": str(
            provenance / "checkpoint-tokenizer-sha256.json"
        ),
        "model_cards": str(provenance / "installed-model-card-sha256.json"),
        "installed_packages": str(provenance / "installed-packages.txt"),
        "scripts": str(provenance / "final-evaluation-scripts-sha256.json"),
    },
}
write(RESULTS / "final-quality-coverage.json", global_report)

# Last task-owned evidence write; root may retain/hash and clean up the rental.
write(
    ROOT / "FINAL_RESULTS_READY.json",
    {
        "status": (
            "ASR evidence complete; no more task-owned ASR writes or GPU inference "
            "planned"
        ),
        "completed_utc": global_report["completed_utc"],
        "rental_instance_id": 59047568,
        "quality_coverage": str(RESULTS / "final-quality-coverage.json"),
        "quality_coverage_sha256": digest(RESULTS / "final-quality-coverage.json"),
        "first_pass_index": str(RESULTS / "index.json"),
        "corrected_pass_index": str(RESULTS / "corrected/index.json"),
        "audio_preservation": str(RESULTS / "corrected/paired-audio-samples.json"),
        "asset_hash_manifest": str(provenance / "checkpoint-tokenizer-sha256.json"),
        "asr_gpu_passes_completed": 2,
    },
)
print(
    json.dumps(
        {
            "logical_audio_cases": len(all_cases),
            "distinct_audio_hashes": len({x["sha256"] for x in all_cases}),
            "http_status_counts": dict(totals),
            "review_flagged_logical_cases": global_report[
                "total_review_flagged_logical_cases"
            ],
            "marker": str(ROOT / "FINAL_RESULTS_READY.json"),
        },
        indent=2,
    )
)
