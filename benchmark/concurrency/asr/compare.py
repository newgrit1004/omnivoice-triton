#!/usr/bin/env python3
"""Pair saved before/after WAV quality screens without hiding missing outputs.

Input is evaluate.py scored.json from each arm. Pairing alone does not enforce
determinism; verify startup-seed evidence and original WAV \
hashes. A negative delta means
fewer ASR edits, not automatically improved human-perceived quality.
"""

import argparse
import json
from pathlib import Path


def load(path):
    rows = json.loads(Path(path).read_text())
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(by_id):
        raise ValueError("Duplicate IDs in quality screen")
    return by_id


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
before, after = load(args.baseline), load(args.candidate)


def asr_hardware(scored):
    files = sorted(Path(scored).parent.glob("*-metadata.json"))
    if not files:
        return {"physical_gpu_nvidia_smi": "unverified: metadata missing"}
    meta = json.loads(files[0].read_text())
    return {
        k: meta.get(k)
        for k in [
            "physical_gpu_nvidia_smi",
            "torch_visible_gpu",
            "cuda_runtime",
            "date_utc",
            "co_resident_gpu_processes",
            "packages",
        ]
    }


hardware = {
    "baseline_asr": asr_hardware(args.baseline),
    "candidate_asr": asr_hardware(args.candidate),
    "tts_hardware": "unverified by ASR comparison; use each TTS run metadata",
}
common = sorted(before.keys() & after.keys())
pairs = []
for case_id in common:
    b, a = before[case_id], after[case_id]
    if b["text"] != a["text"] or b["lang_id"] != a["lang_id"]:
        raise ValueError(f"Corpus/language changed in paired case {case_id}")
    bj = {j["model_card"]: j for j in b["judges"]}
    aj = {j["model_card"]: j for j in a["judges"]}
    if bj.keys() != aj.keys():
        raise ValueError(f"ASR judge set changed for {case_id}")
    comparison = {}
    for model in bj:
        x, y = bj[model], aj[model]
        comparison[model] = {
            "before_transcript": x["transcript"],
            "after_transcript": y["transcript"],
            "before_wer": x["wer"],
            "after_wer": y["wer"],
            "wer_delta": y["wer"] - x["wer"],
            "before_cer": x["cer"],
            "after_cer": y["cer"],
            "cer_delta": y["cer"] - x["cer"],
            "before_deletions": x["deletions"],
            "after_deletions": y["deletions"],
            "before_word_edits": x["substitutions"] + x["deletions"] + x["insertions"],
            "after_word_edits": y["substitutions"] + y["deletions"] + y["insertions"],
        }
    pairs.append(
        {
            "id": case_id,
            "language": a["lang_id"],
            "expected": a["text"],
            "before_audio": b["path"],
            "after_audio": a["path"],
            "before_sha256": b["sha256"],
            "after_sha256": a["sha256"],
            "identical_wav_sha256": b["sha256"] == a["sha256"],
            "before_flags": b["flags"],
            "after_flags": a["flags"],
            "judges": comparison,
            "any_judge_worse": any(
                j["after_word_edits"] > j["before_word_edits"]
                or j["after_cer"] > j["before_cer"]
                for j in comparison.values()
            ),
            "any_judge_more_deletions": any(
                j["after_deletions"] > j["before_deletions"]
                for j in comparison.values()
            ),
        }
    )
missing_before = sorted(after.keys() - before.keys())
missing_after = sorted(before.keys() - after.keys())
summary = {
    "hardware": hardware,
    "baseline_audio_cases": len(before),
    "candidate_audio_cases": len(after),
    "paired_cases": len(pairs),
    "missing_before_ids": missing_before,
    "missing_after_ids": missing_after,
    "bitwise_equal_original_wav_pairs": sum(x["identical_wav_sha256"] for x in pairs),
    "bitwise_different_original_wav_pairs": sum(
        not x["identical_wav_sha256"] for x in pairs
    ),
    "paired_any_judge_worse": sum(x["any_judge_worse"] for x in pairs),
    "paired_any_judge_more_deletions": sum(
        x["any_judge_more_deletions"] for x in pairs
    ),
    "candidate_flagged_all_cases": sum(bool(x["flags"]) for x in after.values()),
    "note": (
        "Missing IDs can reflect non-200 HTTP outcomes or incomplete ASR coverage; "
        "join request logs and never drop them from availability denominators. The API "
        "has no per-request seed. Verified startup seeding plus matched sequential "
        "order may produce identical WAVs; inspect server-seed evidence and reported "
        "SHA equality. Unseeded or concurrently scheduled samples remain stochastic. "
        "ASR flags require review and are not proven TTS defects."
    ),
    "status": "review_required"
    if not pairs
    or missing_before
    or missing_after
    or any(x["any_judge_worse"] for x in pairs)
    or any(x["flags"] for x in after.values())
    else "no_asr_screen_regression_observed",
    "per_language": {},
}
for language in sorted({x["language"] for x in pairs}):
    selected = [x for x in pairs if x["language"] == language]
    summary["per_language"][language] = {
        "paired_cases": len(selected),
        "any_judge_worse": sum(x["any_judge_worse"] for x in selected),
        "candidate_flagged": sum(bool(x["after_flags"]) for x in selected),
    }
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
(out / "comparison.json").write_text(
    json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=2)
    + "\n"
)
lines = [
    "Paired OmniVoice audio quality screen",
    "",
    (
        f"Baseline ASR GPU (physical VRAM/driver from nvidia-smi): "
        f"{hardware['baseline_asr']['physical_gpu_nvidia_smi']}"
    ),
    "",
    (
        f"Candidate ASR GPU (physical VRAM/driver from nvidia-smi): "
        f"{hardware['candidate_asr']['physical_gpu_nvidia_smi']}"
    ),
    "",
    (
        "ASR runtime, visible VRAM, date and co-resident processes are retained in "
        "comparison.json. TTS hardware is unverified by this comparison; use the "
        "per-arm TTS benchmark metadata."
    ),
    "",
    (
        f"Status: **{summary['status']}**. {len(pairs)} paired outputs; "
        f"{summary['paired_any_judge_worse']} pairs worsened under at least one judge; "
        f"{summary['candidate_flagged_all_cases']} candidate outputs require review."
    ),
    "",
    (
        f"Identical original WAV SHA256: "
        f"{summary['bitwise_equal_original_wav_pairs']}/{len(pairs)} paired files. "
        f"Startup-seeded sequential qualification, when supplied, is distinct from "
        f"per-request seeding; the HTTP API exposes no per-request seed."
    ),
    "",
    summary["note"],
    "",
    (
        "Both Omnilingual variants share a model family. This is a word/correctness "
        "screen, not a listening study or speaker quality score."
    ),
    "",
]
for case in pairs:
    if (
        not case["before_flags"]
        and not case["after_flags"]
        and not case["any_judge_worse"]
    ):
        continue
    lines.extend(
        [
            f"**{case['id']} ({case['language']})**",
            "",
            f"Expected: {case['expected']}",
            "",
            (
                f"[Before WAV]({case['before_audio']}) · [After "
                f"WAV]({case['after_audio']})"
            ),
            "",
        ]
    )
    for model, judge in case["judges"].items():
        lines.extend(
            [
                (
                    f"{model}: WER {judge['before_wer']:.3f} → "
                    f"{judge['after_wer']:.3f}, CER {judge['before_cer']:.3f} → "
                    f"{judge['after_cer']:.3f}, deletions {judge['before_deletions']} "
                    f"→ {judge['after_deletions']}."
                ),
                "",
                f"Before: {judge['before_transcript']}",
                "",
                f"After: {judge['after_transcript']}",
                "",
            ]
        )
(out / "review.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
