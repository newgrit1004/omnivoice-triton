#!/usr/bin/env python3
"""Run second small ASR pass only after corrected TTS timing releases the GPU."""

import json
import subprocess
import sys
import time
from pathlib import Path

root = Path("/workspace/omnivoice-speedups")
marker = root / "corrected-timing-complete.json"
status = root / "asr-corrected-status.json"


def update(state, **extra):
    row = {
        "state": state,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    status.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


update("waiting_for_gpu_release", required_marker=str(marker))
started = time.monotonic()
while not marker.exists():
    if time.monotonic() - started > 900:
        update(
            "failed",
            reason=(
                "Corrected GPU release marker did not arrive within fifteen minutes; "
                "no model loaded"
            ),
        )
        raise SystemExit(2)
    time.sleep(2)
release = json.loads(marker.read_text())
if release.get("last_exit_code") != 0:
    update(
        "failed", reason="Corrected TTS run failed; GPU not released", release=release
    )
    raise SystemExit(2)
update("running", release=release)
result = subprocess.run(
    [
        sys.executable,
        str(Path(__file__).parent / "run_all.py"),
        "--root",
        str(root),
        "--output",
        str(root / "asr-results/corrected"),
        "--include-arms",
        "before-seeded-critical-corrected",
        "cache32-seeded-critical-corrected",
        "grouped-decode8-corrected",
    ]
)
update("completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
raise SystemExit(result.returncode)
