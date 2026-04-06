"""OmniVoice Triton kernel fusion for inference acceleration.

This package provides Triton-fused GPU kernels (RMSNorm, SwiGLU,
Fused Norm+Residual) and model runner classes that transparently patch
the OmniVoice Qwen3-0.6B backbone for faster inference on CUDA hardware.

Four inference modes are available:

- **Base**: Standard PyTorch inference.
- **Triton**: Triton kernel fusion (RMSNorm, SwiGLU, Fused Norm+Residual).
- **Faster**: CUDA Graph capture and replay (2.3x speedup).
- **Hybrid**: Triton + CUDA Graph combined (2.8x speedup).

Typical usage::

    from omnivoice_triton import create_runner

    runner = create_runner("hybrid")
    runner.load_model()
    result = runner.generate(text="Hello, world!")
    runner.unload_model()
"""

import warnings

__version__ = "0.1.0"


def _check_torch() -> None:
    """Verify PyTorch with CUDA support is installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "omnivoice-triton requires PyTorch with CUDA support. "
            "Install it first:\n"
            "  pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. omnivoice-triton requires a CUDA-capable GPU.",
            stacklevel=2,
        )


_check_torch()

from omnivoice_triton.kernels import (  # noqa: E402
    TritonFusedAddRMSNorm,
    TritonRMSNorm,
    TritonSwiGLU,
    triton_fused_add_rms_norm,
    triton_rms_norm,
    triton_swiglu_forward,
)
from omnivoice_triton.models import (  # noqa: E402
    ALL_RUNNER_NAMES,
    BaseRunner,
    FasterRunner,
    TritonFasterRunner,
    TritonRunner,
    apply_triton_kernels,
    create_runner,
    get_runner_class,
)

__all__ = [
    "__version__",
    # Kernels
    "TritonRMSNorm",
    "TritonSwiGLU",
    "TritonFusedAddRMSNorm",
    "triton_rms_norm",
    "triton_swiglu_forward",
    "triton_fused_add_rms_norm",
    # Models
    "BaseRunner",
    "TritonRunner",
    "FasterRunner",
    "TritonFasterRunner",
    "apply_triton_kernels",
    "get_runner_class",
    "create_runner",
    "ALL_RUNNER_NAMES",
]
