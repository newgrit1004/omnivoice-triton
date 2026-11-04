#!/usr/bin/env python3
"""Strict fixed-seed iterative-scoring experiment, outside production source.

Run only while the rented GPU has no timed inference or ASR workload. The
prototype keeps per-row position RNG calls and exact target lengths. It falls
back to the original function for mixed lengths or stochastic class sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from omnivoice import OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample
from run import collect_hardware, percentile, write_json


def grouped_iterative(model, task, config, *, original):
    """Explicit upstream0.1.4 algorithm with only deterministic scoring grouped."""
    batch_size = task.batch_size
    if config.class_temperature != 0 or len(set(task.target_lens)) != 1:
        return original(task, config)
    inputs = [
        model._prepare_inference_inputs(
            task.texts[i],
            task.target_lens[i],
            task.ref_texts[i],
            task.ref_audio_tokens[i],
            task.langs[i],
            task.instructs[i],
            config.denoise,
        )
        for i in range(batch_size)
    ]
    c_lens = [item["input_ids"].size(2) for item in inputs]
    max_c_len = max(c_lens)
    target_len = task.target_lens[0]
    mask_id = model.config.audio_mask_id
    codebooks = model.config.num_audio_codebook
    batch_ids = torch.full(
        (2 * batch_size, codebooks, max_c_len),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    audio_mask = torch.zeros(
        (2 * batch_size, max_c_len),
        dtype=torch.bool,
        device=model.device,
    )
    attention_mask = torch.zeros(
        (2 * batch_size, 1, max_c_len, max_c_len),
        dtype=torch.bool,
        device=model.device,
    )
    for i, item in enumerate(inputs):
        c_len = c_lens[i]
        batch_ids[i, :, :c_len] = item["input_ids"]
        audio_mask[i, :c_len] = item["audio_mask"]
        attention_mask[i, :, :c_len, :c_len] = True
        batch_ids[batch_size + i, :, :target_len] = item["input_ids"][..., -target_len:]
        audio_mask[batch_size + i, :target_len] = item["audio_mask"][..., -target_len:]
        attention_mask[batch_size + i, :, :target_len, :target_len] = True
        if max_c_len > target_len:
            diagonal = torch.arange(target_len, max_c_len, device=model.device)
            attention_mask[batch_size + i, :, diagonal, diagonal] = True
    tokens = torch.full(
        (batch_size, codebooks, target_len),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=config.num_step + 1,
        t_shift=config.t_shift,
    ).tolist()
    total_mask = target_len * codebooks
    remaining = total_mask
    schedule = []
    for step in range(config.num_step):
        count = (
            remaining
            if step == config.num_step - 1
            else min(
                math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                remaining,
            )
        )
        schedule.append(int(count))
        remaining -= int(count)
    layer_ids = torch.arange(codebooks, device=model.device).view(1, -1, 1)
    for step in range(config.num_step):
        logits = model(
            input_ids=batch_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
        ).logits
        count = schedule[step]
        if count <= 0:
            continue
        # The FP32 cast is elementwise and only target positions are used below.
        # Prefix/padding logits remain in their model dtype.
        if batch_size == 1:
            c_logits = logits[:1, :, c_lens[0] - target_len : c_lens[0], :].float()
        else:
            c_logits = torch.stack(
                [
                    logits[i, :, c_len - target_len : c_len, :]
                    for i, c_len in enumerate(c_lens)
                ]
            ).float()
        u_logits = logits[batch_size:, :, :target_len, :].float()
        predicted, confidence = model._predict_tokens_with_scoring(
            c_logits,
            u_logits,
            config,
        )
        for i in range(batch_size):
            scores = confidence[i : i + 1] - layer_ids * config.layer_penalty_factor
            if config.position_temperature > 0:
                scores = _gumbel_sample(scores, config.position_temperature)
            sample_tokens = tokens[i : i + 1, :, :target_len]
            scores.masked_fill_(sample_tokens != mask_id, -float("inf"))
            _, topk_idx = torch.topk(scores.flatten(), count)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = predicted[i : i + 1].flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))
            tokens[i : i + 1, :, :target_len] = sample_tokens
            c_len = c_lens[i]
            batch_ids[i : i + 1, :, c_len - target_len : c_len] = sample_tokens
            batch_ids[batch_size + i : batch_size + i + 1, :, :target_len] = (
                sample_tokens
            )
    return [tokens[i, :, :target_len] for i in range(batch_size)]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument(
        "--references", type=Path, default=Path(__file__).parent / "inputs/references"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-sizes", default="1,2,8")
    parser.add_argument("--languages", default="en,es,ar")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from omnivoice_triton import create_runner

    runner = create_runner(
        "hybrid",
        device="cuda",
        model_id=str(args.model),
        dtype="fp16",
        max_cuda_graphs=8,
    )
    runner.load_model()
    model = runner.model
    model.eval()
    original = model._generate_iterative
    predictor = model._predict_tokens_with_scoring
    config = OmniVoiceGenerationConfig(
        num_step=args.steps,
        class_temperature=0.0,
        position_temperature=5.0,
        postprocess_output=False,
    )
    hardware = collect_hardware()
    write_json(
        args.output / "metadata.json",
        {
            "hardware": hardware,
            "source_ref": args.source_ref,
            "model": str(args.model),
            "model_revision": args.model_revision,
            "prototype_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workload": (
                f"fixed-seed hybrid OmniVoice0.1.4 iterative token inference; EN/ES/AR "
                f"cloned references; equal target lengths per batch; {args.steps} "
                f"steps, class_temperature0, position_temperature5; codec decoding "
                f"excluded"
            ),
            "co_resident_load": (
                "GPU exclusive required; collected nvidia-smi process evidence"
            ),
            "qualification": (
                "Requires bitwise token AND per-step confidence equality, not "
                "approximate waveform similarity"
            ),
        },
    )
    references = json.loads((args.references / "manifest.json").read_text())
    texts = {
        "en": "Hello! How are you feeling today?",
        "es": "Hola, ¿cómo te sientes hoy?",
        "ar": "مرحبا، كيف تشعر اليوم؟",
    }
    summary = []
    with torch.inference_mode():
        for entry in references:
            language = entry["lang"]
            if language not in args.languages.split(","):
                continue
            model_language = entry.get("language", language)
            path = args.references / entry["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(path),
                ref_text=entry["text"],
                preprocess_prompt=True,
            )
            for batch_size in map(int, args.batch_sizes.split(",")):
                case = f"{language}-b{batch_size}"
                output_dir = args.output / case
                output_dir.mkdir()
                task = model._preprocess_all(
                    text=[texts[language]] * batch_size,
                    language=[model_language] * batch_size,
                    ref_text=None,
                    ref_audio=None,
                    voice_clone_prompt=[prompt] * batch_size,
                    instruct=None,
                    preprocess_prompt=True,
                    speed=[1.0] * batch_size,
                    duration=None,
                )
                assert len(set(task.target_lens)) == 1
                functions = {
                    "original": original,
                    "grouped": lambda t, c: grouped_iterative(
                        model, t, c, original=original
                    ),
                }
                for function in functions.values():
                    set_seed(100)
                    function(task, config)
                token_outputs, score_outputs, pred_outputs = {}, {}, {}
                for arm, function in functions.items():
                    scores, predictions = [], []

                    def recording_predict(self, c_logits, u_logits, generation_config):
                        predicted, confidence = predictor(
                            c_logits, u_logits, generation_config
                        )
                        for row in range(predicted.shape[0]):
                            predictions.append(predicted[row].clone())
                            scores.append(confidence[row].clone())
                        return predicted, confidence

                    model._predict_tokens_with_scoring = MethodType(
                        recording_predict, model
                    )
                    set_seed(20261103)
                    tokens = function(task, config)
                    torch.cuda.synchronize()
                    token_outputs[arm] = torch.stack(tokens).cpu()
                    score_outputs[arm] = torch.stack(scores).cpu()
                    pred_outputs[arm] = torch.stack(predictions).cpu()
                    np.save(
                        output_dir / (arm + "-tokens.npy"), token_outputs[arm].numpy()
                    )
                    np.save(
                        output_dir / (arm + "-confidence.npy"),
                        score_outputs[arm].numpy(),
                    )
                    model._predict_tokens_with_scoring = predictor
                score_difference = score_outputs["grouped"] - score_outputs["original"]
                comparisons = {
                    "tokens_bitwise_equal": torch.equal(
                        token_outputs["original"], token_outputs["grouped"]
                    ),
                    "confidence_bitwise_equal": torch.equal(
                        score_outputs["original"], score_outputs["grouped"]
                    ),
                    "predicted_tokens_bitwise_equal": torch.equal(
                        pred_outputs["original"], pred_outputs["grouped"]
                    ),
                    "confidence_max_abs_difference": float(
                        score_difference.abs().max()
                    ),
                    "changed_token_count": int(
                        (token_outputs["original"] != token_outputs["grouped"]).sum()
                    ),
                    "target_lengths": task.target_lens,
                }
                timing_rows = []
                for repetition in range(args.repetitions):
                    arms = list(functions)
                    if repetition % 2:
                        arms.reverse()
                    for arm in arms:
                        set_seed(20261103)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        functions[arm](task, config)
                        torch.cuda.synchronize()
                        timing_rows.append(
                            {
                                "arm": arm,
                                "repetition": repetition,
                                "elapsed_ms": (time.perf_counter() - started) * 1000,
                                "peak_allocated_bytes": (
                                    # Preserve allocation beside its timing sample.
                                    torch.cuda.max_memory_allocated()
                                ),
                            }
                        )
                original_ms = percentile(
                    [r["elapsed_ms"] for r in timing_rows if r["arm"] == "original"], 50
                )
                grouped_ms = percentile(
                    [r["elapsed_ms"] for r in timing_rows if r["arm"] == "grouped"], 50
                )
                row = {
                    "case": case,
                    "language": language,
                    "model_language": model_language,
                    "batch_size": batch_size,
                    "comparisons": comparisons,
                    "original_p50_ms": original_ms,
                    "grouped_p50_ms": grouped_ms,
                    "speedup": original_ms / grouped_ms,
                }
                summary.append(row)
                write_json(output_dir / "comparisons.json", comparisons)
                write_json(output_dir / "timings.json", timing_rows)
                write_json(args.output / "summary.json", summary)
                print(json.dumps(row), flush=True)
    runner.unload_model()


if __name__ == "__main__":
    main()
