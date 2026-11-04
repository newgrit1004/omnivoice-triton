#!/usr/bin/env python3
"""Complete-WAV loopback benchmark; save all successful audio for blind ASR.

GPU model/VRAM/runtime are collected at execution, never assumed from the offer.
This program does not claim that HTTP success establishes intelligible speech.
No paid inference API is called. No expected text is supplied to an ASR model.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
LANGUAGES = {"en": "eng_Latn", "es": "spa_Latn", "ar": "arb_Arab"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation, matching NumPy's default percentile definition."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(rows: list[dict[str, Any]], phase_wall_s: float) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == 200]
    latencies = [row["complete_ms"] for row in success]
    return {
        "attempts": len(rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "http200": len(success),
        "signal_valid": sum(
            row.get("audio", {}).get("signal_valid", False) for row in rows
        ),
        "completed_within_500ms": sum(row["complete_ms"] <= 500 for row in success),
        "http200_p50_ms": percentile(latencies, 50),
        "http200_p95_ms": percentile(latencies, 95),
        "http200_max_ms": max(latencies) if latencies else None,
        "all_status_p95_ms": percentile([row["complete_ms"] for row in rows], 95),
        "phase_wall_s": phase_wall_s,
        "http200_per_wall_s": len(success) / phase_wall_s if phase_wall_s else None,
        "quality_status": (
            "ASR and listening review pending; signal checks are not word checks"
        ),
        "latency_denominator": (
            "HTTP 200 only; rejected/failed/timed-out attempts are counted separately"
        ),
    }


def collect_hardware() -> dict[str, Any]:
    import importlib.metadata
    import platform

    import torch

    def command(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, text=True, stderr=subprocess.STDOUT, timeout=15
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unverified: {exc}"

    packages = {}
    for package in [
        "torch",
        "torchaudio",
        "triton",
        "transformers",
        "omnivoice",
        "fastapi",
        "starlette",
        "anyio",
        "uvicorn",
        "httpx",
        "numpy",
        "soundfile",
    ]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": prop.name,
                "visible_vram_bytes": prop.total_memory,
                "compute_capability": [prop.major, prop.minor],
            }
        )
    return {
        "collected_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "cpu": command(["lscpu"]),
        "packages": packages,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_devices": devices,
        "nvidia_smi_gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.used,driver_version,power.limit",
                "--format=csv",
            ]
        ),
        "nvidia_smi_processes": command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv",
            ]
        ),
        "physical_vram_evidence": (
            "nvidia-smi memory.total; visible memory independently from torch device "
            "properties"
        ),
    }


def source_identity(source: Path) -> dict[str, Any]:
    source_hash = hashlib.sha256()
    files = []
    for path in sorted((source / "src").rglob("*.py")):
        relative = str(path.relative_to(source))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hash.update(relative.encode() + b"\0" + digest.encode() + b"\n")
        files.append({"path": relative, "sha256": digest})
    try:
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.SubprocessError:
        head = "archive; see --source-ref and source digest"
    return {
        "source_ref": head,
        "source_tree_sha256": source_hash.hexdigest(),
        "files": files,
    }


async def wait_ready(
    url: str, process: subprocess.Popen, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    (
                        f"Server exited with status {process.returncode}; inspect "
                        f"server.log"
                    )
                )
            try:
                ready = await client.get(url + "/healthz")
                if ready.status_code == 200:
                    return (await client.get(url + "/health")).json()
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Server readiness deadline exceeded; inspect server.log")


async def register_references(
    client: httpx.AsyncClient, url: str, inputs: Path, output: Path
) -> dict[str, str]:
    registrations, prompt_ids = [], {}
    refs = json.loads((inputs / "references/manifest.json").read_text())
    for reference in refs:
        path = inputs / "references" / reference["file"]
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == reference["sha256"], (
            "Reference audio hash mismatch"
        )
        started = time.perf_counter()
        response = await client.post(
            url + "/clone-prompts",
            data={"ref_text": reference["text"], "preprocess_prompt": "true"},
            files={"ref_audio": (path.name, body, "audio/wav")},
        )
        row = {
            "language": reference["lang"],
            "reference": reference,
            "status": response.status_code,
            "complete_ms": (time.perf_counter() - started) * 1000,
            "response": response.json(),
        }
        registrations.append(row)
        write_json(output / "registrations.json", registrations)
        response.raise_for_status()
        prompt_ids[reference["lang"]] = response.json()["prompt_id"]
    return prompt_ids


async def benchmark(args: argparse.Namespace, process: subprocess.Popen) -> None:
    output, url = args.output, f"http://127.0.0.1:{args.port}"
    ready = await wait_ready(url, process, args.startup_timeout)
    write_json(output / "health-before.json", ready)
    if ready.get("max_cuda_graphs") != args.max_cuda_graphs:
        raise RuntimeError("Requested graph limit does not match actual server health")
    if (
        args.audio_decode_batch_size
        and ready.get("audio_decode_batch_size") != args.audio_decode_batch_size
    ):
        raise RuntimeError(
            "Requested codec batch size does not match actual server health"
        )
    if args.startup_seed is not None:
        evidence = json.loads((output / "server-seed.json").read_text())
        if (
            evidence.get("torch_initial_seed") != args.startup_seed
            or evidence.get("cuda_initial_seed") != args.startup_seed
            or not evidence.get("applied_before_module_import")
        ):
            raise RuntimeError("Server did not apply the requested startup seed")
        metadata = json.loads((output / "metadata.json").read_text())
        metadata["seed"].update(applied=True, server_evidence=evidence)
        write_json(output / "metadata.json", metadata)
    corpus = json.loads((args.inputs / "corpus.json").read_text())
    write_json(output / "corpus.json", corpus)
    limits = httpx.Limits(
        max_connections=args.connections,
        max_keepalive_connections=min(128, args.connections),
    )
    phase_summaries, quality_manifest = {}, []
    phase_name = "setup"
    probe_stop = asyncio.Event()

    async def health_probes() -> None:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as health_client:
            while not probe_stop.is_set():
                started = time.perf_counter()
                row: dict[str, Any] = {"at": utc_now(), "phase": phase_name}
                try:
                    response = await health_client.get(url + "/health")
                    row.update(status=response.status_code, body=response.json())
                except (httpx.HTTPError, ValueError) as exc:
                    row.update(status=None, error=repr(exc))
                row["complete_ms"] = (time.perf_counter() - started) * 1000
                append_json(output / "health-probes.jsonl", row)
                try:
                    await asyncio.wait_for(probe_stop.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

    async with httpx.AsyncClient(
        timeout=args.request_timeout, limits=limits, trust_env=False
    ) as client:
        prompt_ids = await register_references(client, url, args.inputs, output)
        probes = asyncio.create_task(health_probes())

        async def request(
            phase: str,
            index: int,
            case: dict[str, Any],
            gate: asyncio.Event,
            scheduled: float,
        ) -> dict[str, Any]:
            await gate.wait()
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            params = {
                "mode": "clone",
                "text": case["text"],
                "language": case["api_language"],
                "prompt_id": prompt_ids[case["language"]],
                "num_step": str(args.steps),
                "speed": "1.0",
                "denoise": "true",
                "postprocess_mode": "off",
                "insert_pauses": "false",
            }
            started = time.perf_counter()
            identity = f"{phase}-{index:04d}-{case['id']}"
            row: dict[str, Any] = {
                "id": identity,
                "arm": args.label,
                "phase": phase,
                "index": index,
                "case": case,
                "parameters": params,
                "started_at": utc_now(),
                "scheduled_lateness_ms": max(0.0, (started - scheduled) * 1000),
            }
            try:
                response = await client.post(url + "/generate", data=params)
                row.update(
                    status=response.status_code,
                    complete_ms=(time.perf_counter() - started) * 1000,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower().startswith("x-omnivoice-")
                    },
                )
                if response.status_code == 200:
                    path = output / "audio" / (identity + ".wav")
                    path.write_bytes(response.content)
                    digest = hashlib.sha256(response.content).hexdigest()
                    row.update(
                        path=str(path.relative_to(output)),
                        sha256=digest,
                        response_bytes=len(response.content),
                    )
                    try:
                        samples, sample_rate = sf.read(
                            io.BytesIO(response.content),
                            dtype="float32",
                            always_2d=True,
                        )
                        finite = bool(np.isfinite(samples).all())
                        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
                        rms = (
                            float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
                        )
                        row["audio"] = {
                            "duration_s": len(samples) / sample_rate,
                            "sample_rate": sample_rate,
                            "channels": samples.shape[1],
                            "finite": finite,
                            "peak": peak,
                            "rms": rms,
                            "clipped_sample_fraction": float(
                                np.mean(np.abs(samples) >= 0.999)
                            )
                            if samples.size
                            else 0.0,
                            "signal_valid": len(samples) > 0
                            and finite
                            and peak > 0.001
                            and rms > 1e-5,
                        }
                    except Exception as exc:
                        row["audio"] = {
                            "signal_valid": False,
                            "decode_error": repr(exc),
                        }
                    quality_manifest.append(
                        {
                            "id": identity,
                            "arm": args.label,
                            "path": str(path.relative_to(output)),
                            "sha256": digest,
                            "language": case["language"],
                            "omnilingual_language": LANGUAGES[case["language"]],
                            "expected": case["text"],
                            "text": case["text"],
                            "phase": phase,
                            "case_id": case["id"],
                            "request": {"text": case["text"], "phase": phase},
                        }
                    )
                else:
                    row["error_body"] = response.text[:1500]
            except Exception as exc:
                row.update(
                    status=None,
                    complete_ms=(time.perf_counter() - started) * 1000,
                    error=repr(exc),
                )
            append_json(output / "requests.jsonl", row)
            return row

        async def phase(
            name: str,
            cases: list[dict[str, Any]],
            interval_s: float = 0.0,
            sequential: bool = False,
        ) -> None:
            nonlocal phase_name
            if args.phases and name not in args.phases.split(","):
                return
            phase_name = name
            started = time.perf_counter()
            gate = asyncio.Event()
            if sequential:
                gate.set()
                rows = []
                for index, case in enumerate(cases):
                    rows.append(
                        await request(name, index, case, gate, time.perf_counter())
                    )
            else:
                tasks = [
                    asyncio.create_task(
                        request(name, index, case, gate, started + index * interval_s)
                    )
                    for index, case in enumerate(cases)
                ]
                gate.set()
                rows = await asyncio.wait_for(
                    asyncio.gather(*tasks), timeout=args.phase_timeout
                )
            phase_summaries[name] = summary(rows, time.perf_counter() - started)
            write_json(output / "summary.json", phase_summaries)
            write_json(
                output / "quality-manifest.json",
                sorted(quality_manifest, key=lambda row: row["id"]),
            )
            print(
                json.dumps({"arm": args.label, "phase": name, **phase_summaries[name]}),
                flush=True,
            )
            if any(row.get("status") not in (200, 429) for row in rows):
                raise RuntimeError(
                    (
                        f"Unexpected API outcome in {name}; preserve evidence and "
                        f"inspect server.log"
                    )
                )

        try:
            shorts = [case for case in corpus if case["length"] == "short"]
            english = next(case for case in shorts if case["language"] == "en")
            if args.smoke:
                await phase("smoke", [english], sequential=True)
            else:
                await phase("warmup-single", shorts, sequential=True)
                await phase("warmup-burst10", [english] * 10)
                await phase("single-corpus", corpus, sequential=True)
                await phase("single-short-repeat10", [english] * 10, sequential=True)
                await phase("burst10-short", [english] * 10)
                await phase("burst10-mixed", corpus + [english])
                await phase("burst100-short", [english] * 100)
                await phase("staggered-mixed-0.5rps", corpus * 2, interval_s=2)
                for size in [int(x) for x in args.overload_bursts.split(",") if x]:
                    await phase(f"burst{size}-overload", [english] * size)
        finally:
            phase_name = "final"
            probe_stop.set()
            await probes
            write_json(
                output / "quality-manifest.json",
                sorted(quality_manifest, key=lambda row: row["id"]),
            )
            try:
                health = await client.get(url + "/health", timeout=15)
                write_json(output / "health-after.json", health.json())
            except (httpx.HTTPError, ValueError) as exc:
                write_json(output / "health-after.json", {"error": repr(exc)})


def main() -> None:
    if sys.argv[1:] == ["--collect-hardware"]:
        print(json.dumps(collect_hardware()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-ref",
        required=True,
        help="Exact baseline/candidate commit, or commit + recorded patch digest",
    )
    parser.add_argument(
        "--model", required=True, help="Pinned local checkpoint directory"
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-batch-requests", type=int, default=8)
    parser.add_argument("--max-pending-requests", type=int, default=16)
    parser.add_argument("--max-cuda-graphs", type=int, default=8)
    parser.add_argument("--max-batch-target-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-conditioning-tokens", type=int, default=4096)
    parser.add_argument("--batch-bucket-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--collect-ms", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--phase-timeout", type=float, default=360.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--startup-seed", type=int, default=None)
    parser.add_argument("--connections", type=int, default=128)
    parser.add_argument("--overload-bursts", default="")
    parser.add_argument(
        "--audio-decode-batch-size",
        type=int,
        default=0,
        help="0 omits flag for old source compatibility",
    )
    parser.add_argument(
        "--phases",
        default="",
        help=(
            "Optional comma-separated phase subset; include desired warmups explicitly"
        ),
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    args.inputs = args.inputs.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LINGUA_") and key != "WORKER_ID"
    }
    environment["PYTHONPATH"] = str(args.source / "src")
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["OMNIVOICE_LOG_LEVEL"] = "INFO"
    command = [
        sys.executable,
        "-m",
        "omnivoice_triton.cli.api_server",
        "--model",
        args.model,
        "--runner",
        "hybrid",
        "--device",
        "cuda",
        "--dtype",
        "fp16",
        "--no-asr",
        "--ip",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--batch-collect-ms",
        str(args.collect_ms),
        "--max-batch-requests",
        str(args.max_batch_requests),
        "--max-pending-requests",
        str(args.max_pending_requests),
        "--max-pending-gpu-tasks",
        "4",
        "--max-batch-target-tokens",
        str(args.max_batch_target_tokens),
        "--max-batch-conditioning-tokens",
        str(args.max_batch_conditioning_tokens),
        "--batch-bucket-sizes",
        args.batch_bucket_sizes,
        "--max-cuda-graphs",
        str(args.max_cuda_graphs),
    ]
    if args.audio_decode_batch_size:
        command.extend(["--audio-decode-batch-size", str(args.audio_decode_batch_size)])
    if args.startup_seed is not None:
        command = [
            sys.executable,
            str(HERE / "seeded_server.py"),
            "--seed",
            str(args.startup_seed),
            "--evidence",
            str(args.output / "server-seed.json"),
            *command[3:],
        ]
    # PyTorch device inspection initializes CUDA. Use a short-lived child so the
    # benchmark client does not retain a second CUDA context during inference.
    hardware = json.loads(
        subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--collect-hardware"],
            text=True,
            timeout=60,
        )
    )
    write_json(
        args.output / "metadata.json",
        {
            "started_at": utc_now(),
            "arm": args.label,
            "declared_source_ref": args.source_ref,
            "source": source_identity(args.source),
            "hardware": hardware,
            "protocol_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "seed_wrapper_sha256": hashlib.sha256(
                (HERE / "seeded_server.py").read_bytes()
            ).hexdigest(),
            "model": {"path": args.model, "revision": args.model_revision},
            "command": command,
            "profile": {
                "num_step": args.steps,
                "speed": 1.0,
                "denoise": True,
                "postprocess_mode": "off",
                "insert_pauses": False,
                "return": "one complete WAV",
            },
            "seed": {
                "startup_seed": args.startup_seed,
                "applied": False,
                "note": (
                    "API has no per-request seed. Startup-seeded sequential runs with "
                    "matched inputs/order can be hash-compared; concurrent scheduling "
                    "is not seed-paired."
                ),
            },
            "co_resident_load": (
                "Record nvidia-smi process evidence before model launch. ASR is a "
                "separate later phase, not part of latency measurements."
            ),
            "network": (
                "127.0.0.1 loopback on rental; excludes gateway, external network and "
                "MuseTalk"
            ),
            "reference_note": (
                "Reused original EN/ES/AR reference recordings are 12–15 seconds, "
                "longer than upstream 3–10 second guidance; a challenging fixed "
                "fixture, not a production recommendation."
            ),
            "interpretation": (
                "429 is overload rejection; never count it as successful concurrency "
                "capacity. p95 for successful responses is labeled with its "
                "denominator. This is a finite pilot, not a 5000-call certification."
            ),
        },
    )
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=args.source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            args.output / "server-process.json",
            {"pid": process.pid, "started_at": utc_now()},
        )
        outcome = {"status": "failed", "finished_at": None}
        try:
            asyncio.run(benchmark(args, process))
            outcome["status"] = "completed"
        except BaseException as exc:
            outcome["error"] = repr(exc)
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            outcome.update(finished_at=utc_now(), server_exit_code=process.returncode)
            write_json(args.output / "outcome.json", outcome)


if __name__ == "__main__":
    main()
