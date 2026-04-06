"""Monkey-patch OmniVoice to replace PyTorch ops with Triton kernels."""

import logging
import types

import torch.nn as nn

from omnivoice_triton.kernels.fused_norm_residual import triton_fused_add_rms_norm
from omnivoice_triton.kernels.rms_norm import TritonRMSNorm
from omnivoice_triton.kernels.swiglu import triton_swiglu_forward

logger = logging.getLogger(__name__)


def _get_layer_index(name: str) -> int | None:
    """Extract layer index from a dotted module name.

    Args:
        name: Dotted module path such as ``"model.layers.5.input_layernorm"``.

    Returns:
        The integer layer index, or ``None`` if the module is not inside a
        numbered ``layers`` collection (e.g. final norm).
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _should_patch(name: str, patch_range: tuple[int, int] | None) -> bool:
    """Check whether a module should be patched based on its layer index.

    Args:
        name: Dotted module path.
        patch_range: ``(start, end)`` half-open range of layer indices to
            patch.  ``None`` means patch everything (default behaviour).

    Returns:
        ``True`` if the module should receive a Triton patch.
    """
    if patch_range is None:
        return True
    layer_idx = _get_layer_index(name)
    if layer_idx is None:
        return False
    return patch_range[0] <= layer_idx < patch_range[1]


def _get_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Resolve a dotted module path to its parent module and attribute name."""
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    return model.get_submodule(parts[0]), parts[1]


def _replace_rms_norm(model: nn.Module, name: str, old: nn.Module) -> None:
    """Swap an RMSNorm module in-place for TritonRMSNorm, reusing weights."""
    parent, attr = _get_parent(model, name)
    hidden_size = old.weight.shape[0]
    eps = getattr(old, "variance_epsilon", getattr(old, "eps", 1e-6))

    new_norm = TritonRMSNorm(hidden_size, eps=eps)
    new_norm.weight = old.weight  # share parameter, no copy
    setattr(parent, attr, new_norm)


def _patch_mlp_forward(mlp: nn.Module) -> None:
    """Replace MLP forward method to use the fused Triton SwiGLU kernel."""

    def _forward(self: nn.Module, x):  # type: ignore[override]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(triton_swiglu_forward(gate, up))

    mlp.forward = types.MethodType(_forward, mlp)


def _patch_decoder_layer_forward(layer: nn.Module) -> None:
    """Replace decoder layer forward to fuse residual add and post-attention norm.

    Monkey-patches ``layer.forward`` so that the residual addition and
    ``post_attention_layernorm`` are executed as a single fused Triton kernel.

    Note: This function detects the norm attribute name dynamically to support
    both ``post_attention_layernorm`` (Qwen2-style) and other naming conventions.
    """
    # Detect the post-attention norm attribute name
    post_attn_norm_name = None
    for candidate in ["post_attention_layernorm", "post_self_attn_layernorm"]:
        if hasattr(layer, candidate):
            post_attn_norm_name = candidate
            break

    if post_attn_norm_name is None:
        logger.warning(
            "Cannot find post-attention norm on %s — skipping fused norm patch",
            type(layer).__name__,
        )
        return

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # FUSED: residual add + post_attention_layernorm (1 kernel)
        post_attn_norm = getattr(self, post_attn_norm_name)
        norm_weight = post_attn_norm.weight
        norm_eps = getattr(
            post_attn_norm,
            "eps",
            getattr(post_attn_norm, "variance_epsilon", 1e-6),
        )
        hidden_states, residual = triton_fused_add_rms_norm(
            hidden_states, residual, norm_weight, norm_eps
        )

        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # transformers 5.3.0 Qwen3DecoderLayer returns plain tensor
        return hidden_states

    layer.forward = types.MethodType(_forward, layer)


def _detect_sage_kernel() -> tuple:
    """Select the best SageAttention CUDA kernel for the current GPU.

    Returns:
        (kernel_fn, pv_accum_dtype) or (None, None) if unavailable.
    """
    import torch

    try:
        from sageattention.core import (
            sageattn_qk_int8_pv_fp8_cuda,
            sageattn_qk_int8_pv_fp8_cuda_sm90,
            sageattn_qk_int8_pv_fp16_cuda,
        )
    except ImportError:
        return None, None

    if not torch.cuda.is_available():
        return None, None

    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor

    if arch >= 120:  # Blackwell
        logger.info("SageAttention: SM%d (Blackwell) FP8 kernel", arch)
        return sageattn_qk_int8_pv_fp8_cuda, "fp32+fp32"
    if arch >= 90:  # Hopper
        logger.info("SageAttention: SM%d (Hopper) FP8 kernel", arch)
        return sageattn_qk_int8_pv_fp8_cuda_sm90, "fp32+fp32"
    if arch == 89:  # Ada Lovelace
        logger.info("SageAttention: SM%d (Ada) FP8 kernel", arch)
        return sageattn_qk_int8_pv_fp8_cuda, "fp32+fp32"
    if arch >= 80:  # Ampere
        logger.info("SageAttention: SM%d (Ampere) FP16 kernel", arch)
        return sageattn_qk_int8_pv_fp16_cuda, "fp32"

    logger.warning("SageAttention: SM%d < SM80, not supported", arch)
    return None, None


# Module-level cache for kernel detection
_sage_kernel_cache: list = []


def _get_sage_kernel() -> tuple:
    if not _sage_kernel_cache:
        _sage_kernel_cache.append(_detect_sage_kernel())
    return _sage_kernel_cache[0]


def _patch_attention_sage(attn: nn.Module) -> None:
    """Replace attention with SageAttention using architecture-specific kernels.

    Uses low-level CUDA kernels with GPU-specific selection (Blackwell/Hopper/
    Ada/Ampere). Falls back to SDPA when attention_mask is present, as
    sageattn does not support arbitrary masks.

    Reference: https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS

    Args:
        attn: Qwen3Attention module with q_proj, k_proj, v_proj, o_proj.
    """
    import torch.nn.functional as F

    sage_fn, pv_accum_dtype = _get_sage_kernel()
    if sage_fn is None:
        logger.warning("No SageAttention kernel available, skipping patch")
        return

    def _forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # SageAttention: use when no mask and no KV cache
        use_sage = (
            attention_mask is None and past_key_values is None and sage_fn is not None
        )

        if use_sage:
            original_dtype = query_states.dtype
            target_dtype = self.q_proj.weight.dtype

            q = query_states.to(target_dtype)
            k = key_states.to(target_dtype)
            v = value_states.to(target_dtype)

            # SageAttention handles GQA internally — no repeat_kv needed
            attn_output = sage_fn(
                q,
                k,
                v,
                tensor_layout="HND",
                is_causal=True,
                qk_quant_gran="per_warp",
                pv_accum_dtype=pv_accum_dtype,
            )

            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]

            attn_output = attn_output.to(original_dtype)
        else:
            # Fallback: SDPA when mask is present or KV cache in use
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn.forward = types.MethodType(_forward, attn)


def apply_sage_attention(
    model: nn.Module,
    patch_range: tuple[int, int] | None = None,
) -> int:
    """Replace attention with SageAttention in the model.

    Args:
        model: The nn.Module to patch.
        patch_range: Half-open layer range. None patches all.

    Returns:
        Number of attention modules patched.
    """
    try:
        import sageattention  # noqa: F401
    except ImportError:
        logger.warning(
            "sageattention not installed — skipping attention patches. "
            "Install with: pip install sageattention"
        )
        return 0

    count = 0
    for name, module in model.named_modules():
        if (
            type(module).__name__.endswith("Attention")
            and hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "o_proj")
            and _should_patch(name, patch_range)
        ):
            _patch_attention_sage(module)
            count += 1

    logger.info("SageAttention patched: %d attention modules", count)
    return count


def apply_triton_kernels(
    model: nn.Module,
    enable_fused_norm: bool = True,
    patch_range: tuple[int, int] | None = None,
) -> None:
    """Replace PyTorch ops with Triton kernels in OmniVoice LLM backbone.

    Patches applied:
      1. RMSNorm → TritonRMSNorm
      2. MLP activation → fused SwiGLU kernel
      3. residual + post_attn_norm → fused Triton kernel (if enable_fused_norm)

    Args:
        model: The nn.Module to patch (typically model.llm).
        enable_fused_norm: Enable fused norm+residual kernel. Default True.
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch.  ``None`` (default) patches **all** layers.
    """
    if patch_range is not None:
        start, end = patch_range
        if start < 0 or end <= start:
            raise ValueError(
                f"patch_range must satisfy 0 <= start < end, got ({start}, {end})"
            )

    norm_count = 0
    mlp_count = 0
    fused_count = 0

    modules = list(model.named_modules())

    for name, module in modules:
        cls_name = type(module).__name__

        # 1. Replace any RMSNorm variant with TritonRMSNorm
        if (
            "RMSNorm" in cls_name
            and hasattr(module, "weight")
            and _should_patch(name, patch_range)
        ):
            _replace_rms_norm(model, name, module)
            norm_count += 1

        # 2. Patch SwiGLU MLPs (identified by gate/up/down projections)
        if (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and _should_patch(name, patch_range)
        ):
            _patch_mlp_forward(module)
            mlp_count += 1

    # 3. Fused Norm+Residual (decoder layer forward replacement)
    if enable_fused_norm:
        for _name, module in modules:
            if (
                hasattr(module, "input_layernorm")
                and hasattr(module, "self_attn")
                and hasattr(module, "mlp")
                and _should_patch(_name, patch_range)
                # Check for any post-attention norm variant
                and (
                    hasattr(module, "post_attention_layernorm")
                    or hasattr(module, "post_self_attn_layernorm")
                )
            ):
                _patch_decoder_layer_forward(module)
                fused_count += 1

    if patch_range is None:
        logger.info(
            "Triton patching: %d RMSNorm, %d MLP, %d FusedNormResidual",
            norm_count,
            mlp_count,
            fused_count,
        )
    else:
        logger.info(
            "[Partial] Layers [%d, %d): %d RMSNorm, %d MLP, %d FusedNormResidual",
            patch_range[0],
            patch_range[1],
            norm_count,
            mlp_count,
            fused_count,
        )


def find_patchable_model(model: object) -> nn.Module:
    """Find the underlying nn.Module from a model wrapper.

    OmniVoice wraps a Qwen3ForCausalLM internally as `model.llm`.
    This searches for the nn.Module that contains patchable layers.

    Args:
        model: A model object (may or may not be an nn.Module).

    Returns:
        The underlying nn.Module suitable for Triton patching.

    Raises:
        RuntimeError: If no nn.Module can be found inside the wrapper.
    """
    if isinstance(model, nn.Module):
        return model

    candidates = ["llm", "model", "transformer", "talker", "_model"]
    for attr in candidates:
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            logger.info("Found patchable model at .%s", attr)
            return inner

    for attr in dir(model):
        if attr.startswith("_"):
            continue
        val = getattr(model, attr, None)
        if isinstance(val, nn.Module):
            logger.info("Found patchable model at .%s", attr)
            return val

    raise RuntimeError("Cannot find nn.Module inside the model wrapper.")
