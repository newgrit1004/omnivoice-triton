#!/usr/bin/env python3
"""Stress a running API server with voice-prompt registration during CUDA Graph capture.

The hybrid runner captures one CUDA Graph per new input shape (and evicts old ones
past --max-cuda-graphs), so replies of varying length keep triggering captures on the
batcher thread. This script registers clone prompts in parallel while varied-length
/generate calls run. Before the fix, a capture overlapping prompt registration failed
("operation failed due to a previous error during capture"), the process later hit
"device-side assert triggered" on every request, and /health still answered 200.

Usage:
    python scripts/repro_cuda_graph_capture_race.py --base-url http://127.0.0.1:8002 \\
        --ref-audio ref.wav --ref-text "Transcript of ref.wav." --duration-s 90

Exit code 0 when every request succeeded and /health and /healthz are 2xx afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter
from pathlib import Path

import httpx

WORDS = (
    "the quick brown fox jumps over a lazy dog while we talk about coffee markets "
    "trains music weekends gardens cooking travel family weather books and plans "
    "for tomorrow"
).split()


def _register(
    client: httpx.Client, base: str, audio: bytes, ref_text: str
) -> httpx.Response:
    return client.post(
        f"{base}/clone-prompts",
        files={"ref_audio": ("ref.wav", audio, "audio/wav")},
        data={"ref_text": ref_text, "preprocess_prompt": "true"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--ref-audio", required=True, type=Path)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--prompt-workers", type=int, default=1)
    parser.add_argument("--generate-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    audio = args.ref_audio.read_bytes()
    statuses: Counter[str] = Counter()
    first_errors: list[str] = []
    lock = threading.Lock()
    stop_at = time.monotonic() + args.duration_s

    def record(
        kind: str, response: httpx.Response | None, error: str | None = None
    ) -> None:
        status = str(response.status_code) if response is not None else "exception"
        with lock:
            statuses[f"{kind} {status}"] += 1
            failed = response is None or response.status_code >= 500
            if failed and len(first_errors) < 5:
                detail = error or response.text[:240]
                first_errors.append(
                    f"{time.strftime('%H:%M:%S')} {kind} {status}: {detail}"
                )

    def prompt_worker() -> None:
        with httpx.Client(timeout=120.0) as client:
            while time.monotonic() < stop_at:
                try:
                    record(
                        "clone-prompts", _register(client, base, audio, args.ref_text)
                    )
                except httpx.HTTPError as exc:
                    record("clone-prompts", None, f"{type(exc).__name__}: {exc}")

    with httpx.Client(timeout=120.0) as client:
        # one shared voice registered up front, as Lingua caches a character's prompt
        response = _register(client, base, audio, args.ref_text)
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]

    def generate_worker(index: int) -> None:
        rng = random.Random(args.seed * 100 + index)
        with httpx.Client(timeout=180.0) as client:
            while time.monotonic() < stop_at:
                # 3-40 words: a new sequence length, so often a new CUDA Graph capture
                text = (
                    " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40))) + "."
                )
                try:
                    record(
                        "generate",
                        client.post(
                            f"{base}/generate",
                            data={
                                "mode": "clone",
                                "text": text,
                                "language": "en",
                                "prompt_id": prompt_id,
                                "num_step": "16",
                                "postprocess_mode": "off",
                                "speed": "1.0",
                            },
                        ),
                    )
                except httpx.HTTPError as exc:
                    record("generate", None, f"{type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=prompt_worker) for _ in range(args.prompt_workers)
    ]
    threads += [
        threading.Thread(target=generate_worker, args=(i,))
        for i in range(args.generate_workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with httpx.Client(timeout=30.0) as client:
        responses = {
            path: client.get(f"{base}{path}") for path in ("/health", "/healthz")
        }
    health = {path: response.status_code for path, response in responses.items()}
    try:
        health_body = responses["/health"].json()
    except ValueError:
        health_body = {}
    shapes = health_body.get("cuda_graph_shapes")
    server_errors = sum(
        n for key, n in statuses.items() if not key.split()[-1].startswith(("2", "4"))
    )
    ok = server_errors == 0 and all(200 <= code < 300 for code in health.values())
    print(
        json.dumps(
            {
                "ok": ok,
                "statuses": dict(sorted(statuses.items())),
                "server_errors": server_errors,
                "health": health,
                "cuda_graph_shapes_now": shapes,
                "cuda_fault": health_body.get("cuda_fault"),
                "first_errors": first_errors,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
