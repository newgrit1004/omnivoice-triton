"""Triton kernels for OmniVoice optimization.

This package exposes three fused Triton kernels designed for inference-time
acceleration of the OmniVoice Qwen3-0.6B LLM backbone:

- :class:`TritonRMSNorm` / :func:`triton_rms_norm`: fused RMSNorm (single
  HBM roundtrip, llama fp32 casting mode).
- :class:`TritonSwiGLU` / :func:`triton_swiglu_forward`: fused SwiGLU
  activation (``silu(gate) * up``).
- :class:`TritonFusedAddRMSNorm` / :func:`triton_fused_add_rms_norm`: fused
  residual-add + RMSNorm in one kernel launch.

All kernels are forward-only and optimised for CUDA devices.
"""

from omnivoice_triton.kernels.fused_norm_residual import (
    TritonFusedAddRMSNorm,
    triton_fused_add_rms_norm,
)
from omnivoice_triton.kernels.rms_norm import TritonRMSNorm, triton_rms_norm
from omnivoice_triton.kernels.swiglu import TritonSwiGLU, triton_swiglu_forward

__all__ = [
    "triton_rms_norm",
    "TritonRMSNorm",
    "triton_swiglu_forward",
    "TritonSwiGLU",
    "triton_fused_add_rms_norm",
    "TritonFusedAddRMSNorm",
]
