#!/usr/bin/env python3
"""Save hashes of exact cached ASR weights/tokenizers and installed model cards.

No credentials or environment variables are captured. Run after both ASR models
have downloaded. Store this beside evaluation results for reproducibility.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", default=str(Path.home() / ".cache/fairseq2/assets"))
parser.add_argument("--output", required=True)
parser.add_argument(
    "--cards-only",
    action="store_true",
    help="Weights already hashed by prefetch.py; capture installed model cards only",
)
args = parser.parse_args()
spec = importlib.util.find_spec("omnilingual_asr")
roots = [] if args.cards_only else [Path(args.cache)]
if spec and spec.origin:
    roots.append(Path(spec.origin).parent / "cards")
rows = []
seen = set()
for root in roots:
    if not root.exists():
        continue
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        rows.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        )
Path(args.output).write_text(
    json.dumps(
        {
            "scope": "installed model cards only"
            if args.cards_only
            else "cached assets and installed model cards",
            "roots": list(map(str, roots)),
            "files": rows,
        },
        indent=2,
    )
    + "\n"
)
print({"files_hashed": len(rows), "output": args.output})
