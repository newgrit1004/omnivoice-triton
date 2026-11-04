"""Run concurrent HTTP load tests against the OmniVoice API.

Examples:
    uv run python scripts/load_test_api.py --mode design --concurrency 8
    uv run python scripts/load_test_api.py --mode clone --concurrency 100 \
        --requests 100 --ref-audio clone.wav --ref-text "Reference text"
    uv run python scripts/load_test_api.py --mode clone --concurrency 100 \
        --requests 100 --register-clone-prompt --ref-audio clone.wav \
        --ref-text "Reference text"
    uv run python scripts/load_test_api.py --mode clone --concurrency 32 \
        --requests 128 --ref-audio clone.wav --csv results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import mimetypes
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8002/generate"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_TEXT = {
    "auto": "This is an API load test request.",
    "design": "This is a voice design API load test request.",
    "clone": "This is a voice clone API load test request.",
}
DEFAULT_INSTRUCT = "female, young adult"
DEFAULT_REF_TEXT = "This is a reference transcript for load testing."


def _percentile(values: list[float], percentile: float) -> float:
    """Return a percentile using linear interpolation."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _format_float(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.2f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run concurrent load tests against the OmniVoice API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Full /generate endpoint URL.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "design", "clone"],
        default="design",
        help="Generation mode to exercise.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of in-flight requests.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total number of requests to send. Defaults to --concurrency.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Warmup requests to send before timed measurement.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Request text. Uses a mode-specific default when omitted.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint.",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Voice design / clone instruct string.",
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Reuse a previously registered clone prompt by prompt_id.",
    )
    parser.add_argument(
        "--register-clone-prompt",
        action="store_true",
        default=False,
        help="Register the clone prompt once before the timed run, then reuse it.",
    )
    parser.add_argument(
        "--clone-prompt-url",
        default=None,
        help="Full /clone-prompts endpoint URL. Derived from --url when omitted.",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help="Reference audio file for clone mode.",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Reference transcript for clone mode.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=32,
        help="num_step form field.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="guidance_scale form field.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional speed form field.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration form field.",
    )
    parser.add_argument(
        "--denoise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to send denoise=true/false.",
    )
    parser.add_argument(
        "--preprocess-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to preprocess clone prompts.",
    )
    parser.add_argument(
        "--postprocess-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to postprocess generated audio.",
    )
    parser.add_argument(
        "--postprocess-mode",
        choices=["full", "light", "off"],
        default=None,
        help="Optional named postprocess mode override.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-request results.",
    )
    return parser


def _build_form_data(args: argparse.Namespace) -> dict[str, str]:
    text = args.text or DEFAULT_TEXT[args.mode]
    data = {
        "mode": args.mode,
        "text": text,
        "num_step": str(args.num_step),
        "guidance_scale": str(args.guidance_scale),
        "denoise": str(args.denoise).lower(),
        "preprocess_prompt": str(args.preprocess_prompt).lower(),
    }
    if args.postprocess_mode:
        data["postprocess_mode"] = args.postprocess_mode
    else:
        data["postprocess_output"] = str(args.postprocess_output).lower()
    if args.language:
        data["language"] = args.language
    if args.instruct:
        data["instruct"] = args.instruct
    elif args.mode == "design":
        data["instruct"] = DEFAULT_INSTRUCT
    if args.prompt_id:
        data["prompt_id"] = args.prompt_id
    elif args.ref_text:
        data["ref_text"] = args.ref_text
    elif args.mode == "clone":
        data["ref_text"] = DEFAULT_REF_TEXT
    if args.speed is not None:
        data["speed"] = str(args.speed)
    if args.duration is not None:
        data["duration"] = str(args.duration)
    return data


def _validate_args(args: argparse.Namespace) -> None:
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0.")

    if args.requests is None:
        args.requests = args.concurrency
    if args.requests <= 0:
        raise ValueError("--requests must be > 0.")

    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0.")

    if args.mode == "clone":
        if args.prompt_id and args.register_clone_prompt:
            raise ValueError(
                "--prompt-id cannot be combined with --register-clone-prompt."
            )
        if args.prompt_id is None:
            if args.ref_audio is None:
                raise ValueError(
                    "--ref-audio is required for --mode clone unless "
                    "--prompt-id is used."
                )
            if not args.ref_audio.exists():
                raise ValueError(f"Reference audio not found: {args.ref_audio}")
        elif args.ref_audio is not None:
            raise ValueError(
                "--ref-audio cannot be combined with --prompt-id for generate requests."
            )


def _default_clone_prompt_url(generate_url: str) -> str:
    if generate_url.endswith("/generate"):
        return generate_url[: -len("/generate")] + "/clone-prompts"
    return generate_url.rstrip("/") + "/clone-prompts"


async def _register_clone_prompt(
    *,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    audio_name: str,
    audio_bytes: bytes,
    audio_content_type: str | None,
) -> str:
    clone_prompt_url = args.clone_prompt_url or _default_clone_prompt_url(args.url)
    data = {
        "preprocess_prompt": str(args.preprocess_prompt).lower(),
        "ref_text": args.ref_text or DEFAULT_REF_TEXT,
    }

    response = await client.post(
        clone_prompt_url,
        data=data,
        files={
            "ref_audio": (
                audio_name,
                audio_bytes,
                audio_content_type or "application/octet-stream",
            )
        },
    )
    response.raise_for_status()
    payload = response.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("Clone prompt registration succeeded without prompt_id.")
    logger.info(
        "Registered clone prompt prompt_id=%s duration_ms=%s stored_device=%s",
        prompt_id,
        payload.get("duration_ms", "n/a"),
        payload.get("stored_device", "n/a"),
    )
    return str(prompt_id)


def _extract_result(
    *,
    index: int,
    started_local: datetime,
    finished_local: datetime,
    elapsed_s: float,
    response: httpx.Response,
) -> dict[str, Any]:
    headers = response.headers
    return {
        "index": index,
        "status": response.status_code,
        "local_started_at": started_local.isoformat().replace("+00:00", "Z"),
        "local_finished_at": finished_local.isoformat().replace("+00:00", "Z"),
        "elapsed_s": round(elapsed_s, 6),
        "request_id": headers.get("x-omnivoice-request-id"),
        "started_at": headers.get("x-omnivoice-started-at"),
        "finished_at": headers.get("x-omnivoice-finished-at"),
        "latency_ms": headers.get("x-omnivoice-latency-ms"),
        "peak_vram_gb": headers.get("x-omnivoice-peak-vram-gb"),
        "pre_batch_ms": headers.get("x-omnivoice-pre-batch-ms"),
        "prompt_prepare_ms": headers.get("x-omnivoice-prompt-prepare-ms"),
        "batch_estimate_ms": headers.get("x-omnivoice-batch-estimate-ms"),
        "queue_wait_ms": headers.get("x-omnivoice-queue-wait-ms"),
        "batch_exec_ms": headers.get("x-omnivoice-batch-exec-ms"),
        "response_encode_ms": headers.get("x-omnivoice-response-encode-ms"),
        "generate_total_ms": headers.get("x-omnivoice-generate-total-ms"),
        "prepare_inference_inputs_ms": headers.get(
            "x-omnivoice-prepare-inference-inputs-ms"
        ),
        "iterative_generate_ms": headers.get("x-omnivoice-iterative-generate-ms"),
        "chunked_generate_ms": headers.get("x-omnivoice-chunked-generate-ms"),
        "decode_postprocess_ms": headers.get("x-omnivoice-decode-postprocess-ms"),
        "audio_decode_ms": headers.get("x-omnivoice-audio-decode-ms"),
        "post_process_audio_ms": headers.get("x-omnivoice-post-process-audio-ms"),
        "gpu_sample_count": headers.get("x-omnivoice-gpu-sample-count"),
        "gpu_util_avg_pct": headers.get("x-omnivoice-gpu-util-avg-pct"),
        "gpu_util_peak_pct": headers.get("x-omnivoice-gpu-util-peak-pct"),
        "device_vram_used_gb_avg": headers.get(
            "x-omnivoice-device-vram-used-gb-avg"
        ),
        "device_vram_used_gb_peak": headers.get(
            "x-omnivoice-device-vram-used-gb-peak"
        ),
        "device_vram_util_avg_pct": headers.get(
            "x-omnivoice-device-vram-util-avg-pct"
        ),
        "device_vram_util_peak_pct": headers.get(
            "x-omnivoice-device-vram-util-peak-pct"
        ),
        "batch_requests": headers.get("x-omnivoice-batch-requests"),
        "batch_target_tokens": headers.get("x-omnivoice-batch-target-tokens"),
        "batch_conditioning_tokens": headers.get(
            "x-omnivoice-batch-conditioning-tokens"
        ),
        "batch_max_sequence_length": headers.get(
            "x-omnivoice-batch-max-sequence-length"
        ),
        "batch_lane": headers.get("x-omnivoice-batch-lane"),
        "postprocess_mode": headers.get("x-omnivoice-postprocess-mode"),
        "prompt_source": headers.get("x-omnivoice-prompt-source"),
        "prompt_id": headers.get("x-omnivoice-prompt-id"),
        "audio_duration_s": headers.get("x-omnivoice-audio-duration-s"),
        "rtf": headers.get("x-omnivoice-rtf"),
        "content_bytes": len(response.content),
        "error": "" if response.status_code == 200 else response.text[:500],
    }


async def _run_request(
    *,
    index: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    data: dict[str, str],
    audio_name: str | None,
    audio_bytes: bytes | None,
    audio_content_type: str | None,
) -> dict[str, Any]:
    async with semaphore:
        started_local = datetime.now(UTC)
        started_perf = time.perf_counter()
        try:
            files = None
            if audio_bytes is not None and audio_name is not None:
                files = {
                    "ref_audio": (
                        audio_name,
                        audio_bytes,
                        audio_content_type or "application/octet-stream",
                    )
                }
            response = await client.post(url, data=data, files=files)
            finished_perf = time.perf_counter()
            finished_local = datetime.now(UTC)
            return _extract_result(
                index=index,
                started_local=started_local,
                finished_local=finished_local,
                elapsed_s=finished_perf - started_perf,
                response=response,
            )
        except Exception as exc:
            finished_perf = time.perf_counter()
            finished_local = datetime.now(UTC)
            return {
                "index": index,
                "status": 0,
                "local_started_at": started_local.isoformat().replace("+00:00", "Z"),
                "local_finished_at": finished_local.isoformat().replace("+00:00", "Z"),
                "elapsed_s": round(finished_perf - started_perf, 6),
                "request_id": None,
                "started_at": None,
                "finished_at": None,
                "latency_ms": None,
                "peak_vram_gb": None,
                "pre_batch_ms": None,
                "prompt_prepare_ms": None,
                "batch_estimate_ms": None,
                "queue_wait_ms": None,
                "batch_exec_ms": None,
                "response_encode_ms": None,
                "generate_total_ms": None,
                "prepare_inference_inputs_ms": None,
                "iterative_generate_ms": None,
                "chunked_generate_ms": None,
                "decode_postprocess_ms": None,
                "audio_decode_ms": None,
                "post_process_audio_ms": None,
                "gpu_sample_count": None,
                "gpu_util_avg_pct": None,
                "gpu_util_peak_pct": None,
                "device_vram_used_gb_avg": None,
                "device_vram_used_gb_peak": None,
                "device_vram_util_avg_pct": None,
                "device_vram_util_peak_pct": None,
                "batch_requests": None,
                "batch_target_tokens": None,
                "batch_conditioning_tokens": None,
                "batch_max_sequence_length": None,
                "batch_lane": None,
                "postprocess_mode": None,
                "prompt_source": None,
                "prompt_id": None,
                "audio_duration_s": None,
                "rtf": None,
                "content_bytes": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def _run_load_test(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    data = _build_form_data(args)
    semaphore = asyncio.Semaphore(args.concurrency)

    audio_name: str | None = None
    audio_bytes: bytes | None = None
    audio_content_type: str | None = None
    if args.mode == "clone" and args.ref_audio is not None:
        assert args.ref_audio is not None
        audio_name = args.ref_audio.name
        audio_bytes = args.ref_audio.read_bytes()
        audio_content_type = mimetypes.guess_type(args.ref_audio.name)[0]

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if args.mode == "clone" and args.register_clone_prompt:
            if audio_name is None or audio_bytes is None:
                raise RuntimeError(
                    "--register-clone-prompt requires --ref-audio to prepare "
                    "the prompt."
                )
            prompt_id = await _register_clone_prompt(
                client=client,
                args=args,
                audio_name=audio_name,
                audio_bytes=audio_bytes,
                audio_content_type=audio_content_type,
            )
            args.prompt_id = prompt_id
            data["prompt_id"] = prompt_id
            data.pop("ref_text", None)
            audio_name = None
            audio_bytes = None
            audio_content_type = None

        if args.warmup_requests:
            logger.info("Running %d warmup request(s)...", args.warmup_requests)
            warmup_semaphore = asyncio.Semaphore(1)
            for i in range(args.warmup_requests):
                await _run_request(
                    index=i,
                    client=client,
                    semaphore=warmup_semaphore,
                    url=args.url,
                    data=data,
                    audio_name=audio_name,
                    audio_bytes=audio_bytes,
                    audio_content_type=audio_content_type,
                )

        logger.info(
            "Starting load test: mode=%s requests=%d concurrency=%d url=%s",
            args.mode,
            args.requests,
            args.concurrency,
            args.url,
        )
        wall_started = time.perf_counter()
        tasks = [
            _run_request(
                index=i,
                client=client,
                semaphore=semaphore,
                url=args.url,
                data=data,
                audio_name=audio_name,
                audio_bytes=audio_bytes,
                audio_content_type=audio_content_type,
            )
            for i in range(args.requests)
        ]
        rows = await asyncio.gather(*tasks)
        wall_elapsed_s = time.perf_counter() - wall_started
        return rows, wall_elapsed_s


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    wall_elapsed_s: float,
) -> None:
    successes = [row for row in rows if row["status"] == 200]
    failures = [row for row in rows if row["status"] != 200]
    throughput = len(rows) / wall_elapsed_s if wall_elapsed_s > 0 else float("nan")

    logger.info("")
    logger.info("Load Test Summary")
    logger.info(
        "  mode=%s requests=%d concurrency=%d success=%d failures=%d "
        "wall=%.2fs rate=%.2f req/s",
        args.mode,
        len(rows),
        args.concurrency,
        len(successes),
        len(failures),
        wall_elapsed_s,
        throughput,
    )

    if successes:
        latencies = [float(row["latency_ms"]) for row in successes if row["latency_ms"]]
        pre_batch_times = [
            float(row["pre_batch_ms"])
            for row in successes
            if row["pre_batch_ms"] is not None
        ]
        peak_vram_gb = [
            float(row["peak_vram_gb"])
            for row in successes
            if row["peak_vram_gb"] is not None
        ]
        prompt_prepare_times = [
            float(row["prompt_prepare_ms"])
            for row in successes
            if row["prompt_prepare_ms"] is not None
        ]
        batch_estimate_times = [
            float(row["batch_estimate_ms"])
            for row in successes
            if row["batch_estimate_ms"] is not None
        ]
        queue_waits = [
            float(row["queue_wait_ms"])
            for row in successes
            if row["queue_wait_ms"] is not None
        ]
        batch_execs = [
            float(row["batch_exec_ms"])
            for row in successes
            if row["batch_exec_ms"] is not None
        ]
        response_encode_times = [
            float(row["response_encode_ms"])
            for row in successes
            if row["response_encode_ms"] is not None
        ]
        generate_total_times = [
            float(row["generate_total_ms"])
            for row in successes
            if row["generate_total_ms"] is not None
        ]
        prepare_input_times = [
            float(row["prepare_inference_inputs_ms"])
            for row in successes
            if row["prepare_inference_inputs_ms"] is not None
        ]
        iterative_generate_times = [
            float(row["iterative_generate_ms"])
            for row in successes
            if row["iterative_generate_ms"] is not None
        ]
        chunked_generate_times = [
            float(row["chunked_generate_ms"])
            for row in successes
            if row["chunked_generate_ms"] is not None
        ]
        decode_postprocess_times = [
            float(row["decode_postprocess_ms"])
            for row in successes
            if row["decode_postprocess_ms"] is not None
        ]
        audio_decode_times = [
            float(row["audio_decode_ms"])
            for row in successes
            if row["audio_decode_ms"] is not None
        ]
        post_process_audio_times = [
            float(row["post_process_audio_ms"])
            for row in successes
            if row["post_process_audio_ms"] is not None
        ]
        gpu_util_avg = [
            float(row["gpu_util_avg_pct"])
            for row in successes
            if row["gpu_util_avg_pct"] is not None
        ]
        gpu_util_peak = [
            float(row["gpu_util_peak_pct"])
            for row in successes
            if row["gpu_util_peak_pct"] is not None
        ]
        device_vram_peak = [
            float(row["device_vram_used_gb_peak"])
            for row in successes
            if row["device_vram_used_gb_peak"] is not None
        ]
        device_vram_util_peak = [
            float(row["device_vram_util_peak_pct"])
            for row in successes
            if row["device_vram_util_peak_pct"] is not None
        ]
        batch_sizes = Counter(
            int(row["batch_requests"])
            for row in successes
            if row["batch_requests"] is not None
        )
        prompt_sources = Counter(
            str(row["prompt_source"])
            for row in successes
            if row["prompt_source"] is not None
        )
        postprocess_modes = Counter(
            str(row["postprocess_mode"])
            for row in successes
            if row["postprocess_mode"] is not None
        )

        logger.info(
            "  latency_ms avg=%s p50=%s p95=%s max=%s",
            _format_float(statistics.mean(latencies)),
            _format_float(statistics.median(latencies)),
            _format_float(_percentile(latencies, 0.95)),
            _format_float(max(latencies)),
        )
        if pre_batch_times:
            logger.info(
                "  prebatch_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(pre_batch_times)),
                _format_float(statistics.median(pre_batch_times)),
                _format_float(_percentile(pre_batch_times, 0.95)),
                _format_float(max(pre_batch_times)),
            )
        if peak_vram_gb:
            logger.info(
                "  peak_vram_gb avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(peak_vram_gb)),
                _format_float(statistics.median(peak_vram_gb)),
                _format_float(_percentile(peak_vram_gb, 0.95)),
                _format_float(max(peak_vram_gb)),
            )
        if prompt_prepare_times:
            logger.info(
                "  prompt_ms  avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(prompt_prepare_times)),
                _format_float(statistics.median(prompt_prepare_times)),
                _format_float(_percentile(prompt_prepare_times, 0.95)),
                _format_float(max(prompt_prepare_times)),
            )
        if batch_estimate_times:
            logger.info(
                "  estimate_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(batch_estimate_times)),
                _format_float(statistics.median(batch_estimate_times)),
                _format_float(_percentile(batch_estimate_times, 0.95)),
                _format_float(max(batch_estimate_times)),
            )
        if queue_waits:
            logger.info(
                "  queue_ms   avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(queue_waits)),
                _format_float(statistics.median(queue_waits)),
                _format_float(_percentile(queue_waits, 0.95)),
                _format_float(max(queue_waits)),
            )
        if batch_execs:
            logger.info(
                "  batch_ms   avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(batch_execs)),
                _format_float(statistics.median(batch_execs)),
                _format_float(_percentile(batch_execs, 0.95)),
                _format_float(max(batch_execs)),
            )
        if response_encode_times:
            logger.info(
                "  encode_ms  avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(response_encode_times)),
                _format_float(statistics.median(response_encode_times)),
                _format_float(_percentile(response_encode_times, 0.95)),
                _format_float(max(response_encode_times)),
            )
        if generate_total_times:
            logger.info(
                "  generate_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(generate_total_times)),
                _format_float(statistics.median(generate_total_times)),
                _format_float(_percentile(generate_total_times, 0.95)),
                _format_float(max(generate_total_times)),
            )
        if prepare_input_times:
            logger.info(
                "  prepare_infer_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(prepare_input_times)),
                _format_float(statistics.median(prepare_input_times)),
                _format_float(_percentile(prepare_input_times, 0.95)),
                _format_float(max(prepare_input_times)),
            )
        if iterative_generate_times:
            logger.info(
                "  iterative_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(iterative_generate_times)),
                _format_float(statistics.median(iterative_generate_times)),
                _format_float(_percentile(iterative_generate_times, 0.95)),
                _format_float(max(iterative_generate_times)),
            )
        if chunked_generate_times:
            logger.info(
                "  chunked_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(chunked_generate_times)),
                _format_float(statistics.median(chunked_generate_times)),
                _format_float(_percentile(chunked_generate_times, 0.95)),
                _format_float(max(chunked_generate_times)),
            )
        if decode_postprocess_times:
            logger.info(
                "  decode_ms  avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(decode_postprocess_times)),
                _format_float(statistics.median(decode_postprocess_times)),
                _format_float(_percentile(decode_postprocess_times, 0.95)),
                _format_float(max(decode_postprocess_times)),
            )
        if audio_decode_times:
            logger.info(
                "  audio_decode_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(audio_decode_times)),
                _format_float(statistics.median(audio_decode_times)),
                _format_float(_percentile(audio_decode_times, 0.95)),
                _format_float(max(audio_decode_times)),
            )
        if post_process_audio_times:
            logger.info(
                "  post_audio_ms avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(post_process_audio_times)),
                _format_float(statistics.median(post_process_audio_times)),
                _format_float(_percentile(post_process_audio_times, 0.95)),
                _format_float(max(post_process_audio_times)),
            )
        if gpu_util_avg:
            logger.info(
                "  gpu_avg_pct avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(gpu_util_avg)),
                _format_float(statistics.median(gpu_util_avg)),
                _format_float(_percentile(gpu_util_avg, 0.95)),
                _format_float(max(gpu_util_avg)),
            )
        if gpu_util_peak:
            logger.info(
                "  gpu_peak_pct avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(gpu_util_peak)),
                _format_float(statistics.median(gpu_util_peak)),
                _format_float(_percentile(gpu_util_peak, 0.95)),
                _format_float(max(gpu_util_peak)),
            )
        if device_vram_peak:
            logger.info(
                "  dev_vram_gb avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(device_vram_peak)),
                _format_float(statistics.median(device_vram_peak)),
                _format_float(_percentile(device_vram_peak, 0.95)),
                _format_float(max(device_vram_peak)),
            )
        if device_vram_util_peak:
            logger.info(
                "  dev_vram_pct avg=%s p50=%s p95=%s max=%s",
                _format_float(statistics.mean(device_vram_util_peak)),
                _format_float(statistics.median(device_vram_util_peak)),
                _format_float(_percentile(device_vram_util_peak, 0.95)),
                _format_float(max(device_vram_util_peak)),
            )
        if batch_sizes:
            distribution = ", ".join(
                f"{size}=>{count}" for size, count in sorted(batch_sizes.items())
            )
            logger.info("  per-response batch sizes: %s", distribution)
        if prompt_sources:
            distribution = ", ".join(
                f"{source}=>{count}" for source, count in sorted(prompt_sources.items())
            )
            logger.info("  prompt sources: %s", distribution)
        if postprocess_modes:
            distribution = ", ".join(
                f"{mode}=>{count}" for mode, count in sorted(postprocess_modes.items())
            )
            logger.info("  postprocess modes: %s", distribution)

        server_starts = sorted(
            row["started_at"] for row in successes if row.get("started_at")
        )
        server_finishes = sorted(
            row["finished_at"] for row in successes if row.get("finished_at")
        )
        if server_starts and server_finishes:
            logger.info(
                "  server window: %s -> %s",
                server_starts[0],
                server_finishes[-1],
            )

    if failures:
        logger.info("  failures:")
        for row in failures[:5]:
            logger.info(
                "    index=%s status=%s error=%s",
                row["index"],
                row["status"],
                row["error"],
            )
        if len(failures) > 5:
            logger.info("    ... %d more failure(s)", len(failures) - 5)

    if args.csv is not None:
        logger.info("  csv=%s", args.csv)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    rows, wall_elapsed_s = asyncio.run(_run_load_test(args))

    if args.csv is not None and rows:
        _write_csv(args.csv, rows)

    _summarize(args=args, rows=rows, wall_elapsed_s=wall_elapsed_s)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
