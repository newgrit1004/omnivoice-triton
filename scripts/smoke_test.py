"""Smoke test for OmniVoice model loading and inference.

Verifies:
1. Model loads correctly
2. Model tree structure (attribute names for patching)
3. Basic text-to-speech generation works
4. Voice cloning works
5. Voice design works

Usage:
    uv run python scripts/smoke_test.py
"""

import logging
import time

import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_model_tree(model: object) -> None:
    """Print model tree to identify patchable modules."""
    logger.info("=" * 60)
    logger.info("MODEL TREE INSPECTION")
    logger.info("=" * 60)

    # Top-level attributes
    logger.info("\n--- Top-level attributes ---")
    for attr in dir(model):
        if attr.startswith("_"):
            continue
        val = getattr(model, attr, None)
        if isinstance(val, torch.nn.Module):
            logger.info("  .%s -> %s", attr, type(val).__name__)

    # LLM backbone
    llm = getattr(model, "llm", None)
    if llm is None:
        logger.warning("No .llm attribute found!")
        return

    logger.info("\n--- LLM backbone: %s ---", type(llm).__name__)

    # LLM.model
    inner = getattr(llm, "model", None)
    if inner is None:
        logger.warning("No .llm.model attribute found!")
        return

    logger.info("  .llm.model -> %s", type(inner).__name__)

    # Layers
    layers = getattr(inner, "layers", None)
    if layers is None:
        logger.warning("No .llm.model.layers found!")
        return

    logger.info("  Number of layers: %d", len(layers))

    # Inspect first layer in detail
    layer0 = layers[0]
    logger.info("\n--- Layer 0 structure: %s ---", type(layer0).__name__)
    for name, module in layer0.named_children():
        logger.info("  .%s -> %s", name, type(module).__name__)
        # Sub-modules (e.g., mlp internals)
        for sub_name, sub_module in module.named_children():
            logger.info("    .%s.%s -> %s", name, sub_name, type(sub_module).__name__)
            if hasattr(sub_module, "weight"):
                logger.info("      weight shape: %s", tuple(sub_module.weight.shape))

    # Check critical attribute names for patching
    logger.info("\n--- PATCHING COMPATIBILITY CHECK ---")

    has_input_ln = hasattr(layer0, "input_layernorm")
    has_post_attn_ln = hasattr(layer0, "post_attention_layernorm")
    has_self_attn = hasattr(layer0, "self_attn")
    has_mlp = hasattr(layer0, "mlp")

    logger.info("  input_layernorm: %s", has_input_ln)
    logger.info("  post_attention_layernorm: %s", has_post_attn_ln)
    logger.info("  self_attn: %s", has_self_attn)
    logger.info("  mlp: %s", has_mlp)

    if has_mlp:
        mlp = layer0.mlp
        has_gate = hasattr(mlp, "gate_proj")
        has_up = hasattr(mlp, "up_proj")
        has_down = hasattr(mlp, "down_proj")
        logger.info("  mlp.gate_proj: %s", has_gate)
        logger.info("  mlp.up_proj: %s", has_up)
        logger.info("  mlp.down_proj: %s", has_down)

    # Check all norm attribute names on layer
    logger.info("\n--- All norm-like attributes on layer 0 ---")
    for name, module in layer0.named_modules():
        cls_name = type(module).__name__
        if "Norm" in cls_name or "norm" in name:
            logger.info("  %s -> %s", name or "(self)", cls_name)
            if hasattr(module, "weight"):
                logger.info("    weight shape: %s", tuple(module.weight.shape))
            eps = getattr(
                module,
                "variance_epsilon",
                getattr(module, "eps", "NOT FOUND"),
            )
            logger.info("    eps: %s", eps)

    # Model config summary
    logger.info("\n--- LLM Config ---")
    config = llm.config
    for key in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "rms_norm_eps",
        "rope_theta",
        "head_dim",
    ]:
        val = getattr(config, key, "N/A")
        logger.info("  %s: %s", key, val)


def test_basic_generation(model: object) -> None:
    """Test basic text-to-speech generation."""
    logger.info("\n" + "=" * 60)
    logger.info("BASIC GENERATION TEST")
    logger.info("=" * 60)

    text = "Hello, this is a test of the OmniVoice text to speech system."

    start = time.perf_counter()
    audio_list = model.generate(text=text)
    elapsed = time.perf_counter() - start

    audio = audio_list[0]
    logger.info("  Text: %s", text[:50])
    logger.info("  Audio shape: %s", tuple(audio.shape))
    logger.info("  Duration: %.2f seconds", audio.shape[-1] / 24000)
    logger.info("  Generation time: %.2f seconds", elapsed)
    logger.info("  Peak VRAM: %.2f GB", torch.cuda.max_memory_allocated() / 1024**3)

    torchaudio.save("scripts/output_basic.wav", audio.cpu(), 24000)
    logger.info("  Saved to scripts/output_basic.wav")


def test_voice_design(model: object) -> None:
    """Test voice design generation."""
    logger.info("\n" + "=" * 60)
    logger.info("VOICE DESIGN TEST")
    logger.info("=" * 60)

    text = "안녕하세요, 옴니보이스 음성 합성 시스템 테스트입니다."
    instruct = "female, young adult, high pitch"

    start = time.perf_counter()
    audio_list = model.generate(text=text, instruct=instruct)
    elapsed = time.perf_counter() - start

    audio = audio_list[0]
    logger.info("  Text: %s", text[:50])
    logger.info("  Instruct: %s", instruct)
    logger.info("  Audio shape: %s", tuple(audio.shape))
    logger.info("  Duration: %.2f seconds", audio.shape[-1] / 24000)
    logger.info("  Generation time: %.2f seconds", elapsed)

    torchaudio.save("scripts/output_voice_design.wav", audio.cpu(), 24000)
    logger.info("  Saved to scripts/output_voice_design.wav")


def test_voice_clone(model: object, ref_audio: str, ref_text: str) -> None:
    """Test voice cloning generation."""
    logger.info("\n" + "=" * 60)
    logger.info("VOICE CLONING TEST")
    logger.info("=" * 60)

    text = "This is a voice cloning test using a reference audio sample."

    start = time.perf_counter()
    audio_list = model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    elapsed = time.perf_counter() - start

    audio = audio_list[0]
    logger.info("  Text: %s", text[:50])
    logger.info("  Ref audio: %s", ref_audio)
    logger.info("  Audio shape: %s", tuple(audio.shape))
    logger.info("  Duration: %.2f seconds", audio.shape[-1] / 24000)
    logger.info("  Generation time: %.2f seconds", elapsed)

    torchaudio.save("scripts/output_voice_clone.wav", audio.cpu(), 24000)
    logger.info("  Saved to scripts/output_voice_clone.wav")


def main() -> None:
    """Run all smoke tests."""
    from omnivoice import OmniVoice

    logger.info("Loading OmniVoice model...")
    torch.cuda.reset_peak_memory_stats()

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )

    load_vram = torch.cuda.max_memory_allocated() / 1024**3
    logger.info("Model loaded. VRAM: %.2f GB", load_vram)

    # 1. Model tree inspection (most critical for patching)
    inspect_model_tree(model)

    # 2. Basic generation
    test_basic_generation(model)

    # 3. Voice design
    test_voice_design(model)

    # 4. Voice cloning (skip if no reference audio available)
    # TODO: Update path to actual LJSpeech sample after extraction
    logger.info("\n[Voice cloning test skipped — add ref_audio path to enable]")

    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
