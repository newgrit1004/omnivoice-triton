"""Hybrid OmniVoice runner: Triton kernel fusion + CUDA Graph."""

import logging

from omnivoice_triton.models.faster_runner import FasterRunner
from omnivoice_triton.models.patching import (
    apply_sage_attention,
    apply_triton_kernels,
    find_patchable_model,
)

logger = logging.getLogger(__name__)


class TritonFasterRunner(FasterRunner):
    """FasterRunner with Triton kernel patching applied BEFORE graph capture.

    Combines:
      1. Triton kernel fusion (RMSNorm, SwiGLU, Fused Norm+Residual)
      2. Optional SageAttention
      3. CUDA Graph capture and replay

    Triton patches (and SageAttention) are applied first so that the
    captured graph includes the optimized kernels.

    Args:
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch. Defaults to ``(0, 24)``. ``None`` patches all.
        enable_sage_attention: Replace SDPA with SageAttention. Requires
            ``pip install sageattention``. Gracefully skips if unavailable.
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string.
    """

    def __init__(
        self,
        patch_range: tuple[int, int] | None = (0, 24),
        enable_sage_attention: bool = False,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
    ) -> None:
        super().__init__(device=device, model_id=model_id, dtype=dtype)
        self.patch_range = patch_range
        self.enable_sage_attention = enable_sage_attention

    def load_model(self) -> None:
        """Load model, apply Triton patches, then install CUDA Graph wrapper."""
        # Load base model (without CUDA Graph wrapper yet)
        from omnivoice_triton.models.base_runner import BaseRunner

        BaseRunner.load_model(self)

        # Apply Triton kernel patches FIRST
        patchable = find_patchable_model(self._model)
        apply_triton_kernels(patchable, patch_range=self.patch_range)

        # Apply SageAttention if enabled
        if self.enable_sage_attention:
            apply_sage_attention(patchable, patch_range=self.patch_range)

        # Then install CUDA Graph wrapper on the patched model
        from omnivoice_triton.models.faster_runner import _CUDAGraphForward

        self._graph_forward = _CUDAGraphForward(self._model)
        self._model.forward = self._graph_forward

        logger.info(
            "HybridRunner ready (Triton kernels + CUDA Graph wrapper installed)."
        )
