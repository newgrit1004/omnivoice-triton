"""Per-kernel micro-benchmarks: PyTorch vs Triton.

Benchmarks RMSNorm, SwiGLU, and Fused Norm+Residual kernels
using OmniVoice dimensions (hidden=1024, intermediate=3072).

Usage:
    python -m benchmark.bench_kernels
"""

import json
import logging
from pathlib import Path

import torch
import triton

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# OmniVoice Qwen3-0.6B dimensions
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
BATCH_SEQ = 128  # batch * seq_len tokens
EPS = 1e-6


def _pytorch_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def _pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def _pytorch_fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    s = x + residual
    variance = s.to(torch.float32).pow(2).mean(-1, keepdim=True)
    s_norm = s * torch.rsqrt(variance + eps)
    return s_norm * weight, s


def bench_rms_norm() -> dict:
    """Benchmark RMSNorm: PyTorch vs Triton."""
    from omnivoice_triton.kernels.rms_norm import triton_rms_norm

    x = torch.randn(BATCH_SEQ, HIDDEN_SIZE, device="cuda", dtype=torch.float16)
    w = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.float16)

    pt_ms = triton.testing.do_bench(lambda: _pytorch_rms_norm(x, w, EPS))
    tr_ms = triton.testing.do_bench(lambda: triton_rms_norm(x, w, EPS))

    speedup = pt_ms / tr_ms
    logger.info("RMSNorm: PyTorch %.3fms, Triton %.3fms (%.2fx)", pt_ms, tr_ms, speedup)
    return {
        "kernel": "RMSNorm",
        "pytorch_ms": pt_ms,
        "triton_ms": tr_ms,
        "speedup": speedup,
    }


def bench_swiglu() -> dict:
    """Benchmark SwiGLU: PyTorch vs Triton."""
    from omnivoice_triton.kernels.swiglu import triton_swiglu_forward

    gate = torch.randn(BATCH_SEQ, INTERMEDIATE_SIZE, device="cuda", dtype=torch.float16)
    up = torch.randn(BATCH_SEQ, INTERMEDIATE_SIZE, device="cuda", dtype=torch.float16)

    pt_ms = triton.testing.do_bench(lambda: _pytorch_swiglu(gate, up))
    tr_ms = triton.testing.do_bench(lambda: triton_swiglu_forward(gate, up))

    speedup = pt_ms / tr_ms
    logger.info("SwiGLU: PyTorch %.3fms, Triton %.3fms (%.2fx)", pt_ms, tr_ms, speedup)
    return {
        "kernel": "SwiGLU",
        "pytorch_ms": pt_ms,
        "triton_ms": tr_ms,
        "speedup": speedup,
    }


def bench_fused_norm_residual() -> dict:
    """Benchmark Fused Add+RMSNorm: PyTorch vs Triton."""
    from omnivoice_triton.kernels.fused_norm_residual import triton_fused_add_rms_norm

    x = torch.randn(BATCH_SEQ, HIDDEN_SIZE, device="cuda", dtype=torch.float16)
    r = torch.randn(BATCH_SEQ, HIDDEN_SIZE, device="cuda", dtype=torch.float16)
    w = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.float16)

    pt_ms = triton.testing.do_bench(lambda: _pytorch_fused_add_rms_norm(x, r, w, EPS))
    tr_ms = triton.testing.do_bench(lambda: triton_fused_add_rms_norm(x, r, w, EPS))

    speedup = pt_ms / tr_ms
    logger.info(
        "FusedNormRes: PyTorch %.3fms, Triton %.3fms (%.2fx)", pt_ms, tr_ms, speedup
    )
    return {
        "kernel": "FusedAddRMSNorm",
        "pytorch_ms": pt_ms,
        "triton_ms": tr_ms,
        "speedup": speedup,
    }


def run_kernel_benchmarks() -> list[dict]:
    """Run all kernel benchmarks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        bench_rms_norm(),
        bench_swiglu(),
        bench_fused_norm_residual(),
    ]

    out_path = RESULTS_DIR / "kernel_benchmarks.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    run_kernel_benchmarks()
