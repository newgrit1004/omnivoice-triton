"""Triton-optimized OmniVoice runner."""

import logging

from omnivoice_triton.models.base_runner import BaseRunner
from omnivoice_triton.models.patching import (
    apply_sage_attention,
    apply_triton_kernels,
    find_patchable_model,
)

logger = logging.getLogger(__name__)


class TritonRunner(BaseRunner):
    """BaseRunner with Triton kernel patching applied after model load.

    Replaces RMSNorm, SwiGLU, Norm+Residual ops with fused Triton kernels
    across the Qwen3-0.6B LLM backbone layers. Optionally replaces
    attention with SageAttention.

    Args:
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch.  Defaults to ``(0, 24)`` — keeps last 4 layers
            in PyTorch for pronunciation accuracy.  ``None`` patches all.
        enable_sage_attention: Replace SDPA with SageAttention. Requires
            ``pip install sageattention``. Gracefully skips if unavailable.
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string (``"bf16"``, ``"fp16"``, ``"fp32"``).
    """

    def __init__(
        self,
        patch_range: tuple[int, int] | None = (0, 24),
        enable_sage_attention: bool = False,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
    ) -> None:
        super().__init__(
            device=device,
            model_id=model_id,
            dtype=dtype,
        )
        self.patch_range = patch_range
        self.enable_sage_attention = enable_sage_attention

    def load_model(self) -> None:
        """Load model then apply Triton kernel patches."""
        super().load_model()
        patchable = find_patchable_model(self._model)
        apply_triton_kernels(
            patchable,
            patch_range=self.patch_range,
        )
        if self.enable_sage_attention:
            apply_sage_attention(patchable, patch_range=self.patch_range)
        logger.info("TritonRunner ready (Triton kernels applied).")
