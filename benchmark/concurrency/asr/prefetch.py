#!/usr/bin/env python3
"""Download official ASR weights/tokenizer into fairseq2 cache, no model load.

This performs network/disk I/O only; it does not instantiate a model or execute
GPU inference. Run before timed TTS benchmarking or while TTS setup is idle.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from fairseq2.assets.download_manager import StandardAssetDownloadManager
from fairseq2.utils.uri import Uri

ASSETS = [
    (
        "omniASR_tokenizer_written_v2",
        "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model",
        "tokenizer",
    ),
    (
        "omniASR_LLM_Unlimited_1B_v2",
        "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-Unlimited-1B-v2.pt",
        "model",
    ),
    (
        "omniASR_CTC_1B_v2",
        "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt",
        "model",
    ),
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", default=str(Path.home() / ".cache/fairseq2/assets"))
parser.add_argument("--output", required=True)
args = parser.parse_args()
manager = StandardAssetDownloadManager(Path(args.cache))
rows = []
for name, url, kind in ASSETS:
    started = time.perf_counter()
    method = manager.download_model if kind == "model" else manager.download_tokenizer
    uri = Uri.maybe_parse(url)
    assert uri is not None
    path = method(uri, name, progress=False)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    row = {
        "name": name,
        "url": url,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "download_and_hash_wall_s": time.perf_counter() - started,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
Path(args.output).write_text(
    json.dumps(
        {
            "measurement_kind": "network/disk asset preparation, no GPU inference",
            "assets": rows,
        },
        indent=2,
    )
    + "\n"
)
