#!/usr/bin/env python3
"""Apply an auditable startup seed before loading the TTS package or model."""

import argparse
import json
import runpy
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--probe", action="store_true")
args, remaining = parser.parse_known_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
evidence = {
    "requested_seed": args.seed,
    "torch_initial_seed": torch.initial_seed(),
    "cuda_initial_seed": torch.cuda.initial_seed()
    if torch.cuda.is_available()
    else None,
    "applied_before_module_import": True,
    "torch_version": torch.__version__,
    "note": (
        "Startup seed only; does not enforce deterministic kernels or concurrent "
        "scheduling."
    ),
}
args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
if args.probe:
    print(json.dumps({**evidence, "cpu_random_probe": torch.rand(4).tolist()}))
else:
    import sys

    sys.argv = ["omnivoice-api", *remaining]
    runpy.run_module("omnivoice_triton.cli.api_server", run_name="__main__")
