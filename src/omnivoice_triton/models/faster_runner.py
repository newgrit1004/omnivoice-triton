"""CUDA Graph optimized OmniVoice runner.

Captures the OmniVoice forward pass (embedding + 28-layer LLM + audio heads)
as a CUDA Graph and replays it for each of the N iterative decoding steps,
eliminating kernel launch overhead.
"""

import logging
from collections import OrderedDict
from typing import Any

import torch

from omnivoice_triton.models.base_runner import BaseRunner

logger = logging.getLogger(__name__)

# Cap on per-shape CUDA Graphs kept alive; each pins a private activation pool,
# so an unbounded cache leaks VRAM across many distinct shapes (issue #8).
DEFAULT_MAX_GRAPHS = 8


class _CUDAGraphForward:
    """Wraps OmniVoice.forward() with CUDA Graph capture and replay.

    On first call for a given (seq_len, batch_size) shape:
      1. Warmup forward pass
      2. Capture CUDA Graph
      3. Store static input/output buffers

    On subsequent calls with the same shape:
      1. Copy inputs into static buffers
      2. Replay captured graph
      3. Return output from static buffer

    Graphs are capped at ``max_graphs`` (each owns a private memory pool). Once
    the cap is reached, new shapes run eagerly instead of capturing, keeping
    VRAM bounded across many distinct sequence lengths (issue #8).
    """

    def __init__(self, model: Any, max_graphs: int = DEFAULT_MAX_GRAPHS) -> None:
        if max_graphs < 1:
            raise ValueError(f"max_graphs must be >= 1, got {max_graphs}")
        self._model = model
        self._original_forward = model.forward
        self._max_graphs = max_graphs
        self._graphs: OrderedDict[tuple[int, ...], dict] = OrderedDict()
        self._fallback_warned: set[tuple[int, ...]] = set()

    def _get_shape_key(self, input_ids: torch.Tensor) -> tuple[int, ...]:
        return tuple(input_ids.shape)

    def _capture(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> dict:
        """Capture CUDA Graph for a specific input shape.

        All tensor arguments that may change between replay calls must be
        cloned into static buffers.  The captured graph records GPU addresses,
        so any tensor whose memory is freed/reused after capture will cause
        the graph to read stale data on replay.
        """
        key = self._get_shape_key(input_ids)
        logger.info("Capturing CUDA Graph for shape %s ...", key)

        # Create static buffers for ALL tensor inputs
        static_input_ids = input_ids.clone()
        static_audio_mask = audio_mask.clone()
        static_attn_mask = (
            attention_mask.clone() if attention_mask is not None else None
        )
        static_doc_ids = document_ids.clone() if document_ids is not None else None
        static_pos_ids = position_ids.clone() if position_ids is not None else None

        # Build kwargs from static buffers
        kwargs: dict[str, Any] = {}
        if static_attn_mask is not None:
            kwargs["attention_mask"] = static_attn_mask
        if static_doc_ids is not None:
            kwargs["document_ids"] = static_doc_ids
        if static_pos_ids is not None:
            kwargs["position_ids"] = static_pos_ids

        # Warmup (required before capture)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            static_output = self._original_forward(
                static_input_ids,
                static_audio_mask,
                **kwargs,
            )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture graph — all tensors are static buffers whose addresses
        # remain valid across replay calls.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self._original_forward(
                static_input_ids,
                static_audio_mask,
                **kwargs,
            )

        entry = {
            "graph": graph,
            "static_input_ids": static_input_ids,
            "static_audio_mask": static_audio_mask,
            "static_attn_mask": static_attn_mask,
            "static_doc_ids": static_doc_ids,
            "static_pos_ids": static_pos_ids,
            "static_output": static_output,
        }
        self._graphs[key] = entry
        logger.info("CUDA Graph captured for shape %s", key)
        return entry

    def __call__(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> Any:
        """Graph-aware forward: capture on first call, replay after."""
        # Training mode or labels present → use original forward
        if labels is not None or self._model.training:
            return self._original_forward(
                input_ids,
                audio_mask,
                labels,
                attention_mask,
                document_ids,
                position_ids,
            )

        key = self._get_shape_key(input_ids)
        entry = self._graphs.get(key)

        if entry is None:
            if len(self._graphs) >= self._max_graphs:
                # Cache full: run eagerly instead of capturing, to bound VRAM
                # and avoid re-capture thrash on many distinct shapes (issue #8).
                if key not in self._fallback_warned:
                    self._fallback_warned.add(key)
                    logger.info(
                        "CUDA Graph cache full (%d shapes); running shape %s "
                        "eagerly. Raise max_graphs to cache more.",
                        self._max_graphs,
                        key,
                    )
                return self._original_forward(
                    input_ids,
                    audio_mask,
                    None,
                    attention_mask,
                    document_ids,
                    position_ids,
                )
            entry = self._capture(
                input_ids,
                audio_mask,
                attention_mask,
                document_ids,
                position_ids,
            )

        # Copy ALL inputs into their static buffers before replay
        entry["static_input_ids"].copy_(input_ids)
        entry["static_audio_mask"].copy_(audio_mask)
        if attention_mask is not None and entry["static_attn_mask"] is not None:
            entry["static_attn_mask"].copy_(attention_mask)
        if document_ids is not None and entry["static_doc_ids"] is not None:
            entry["static_doc_ids"].copy_(document_ids)
        if position_ids is not None and entry["static_pos_ids"] is not None:
            entry["static_pos_ids"].copy_(position_ids)

        entry["graph"].replay()

        return entry["static_output"]

    def clear(self) -> None:
        """Release all captured graphs and their private memory pools."""
        for entry in self._graphs.values():
            entry.clear()
        self._graphs.clear()
        self._fallback_warned.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FasterRunner(BaseRunner):
    """BaseRunner with CUDA Graph optimization for the forward pass.

    Captures the OmniVoice forward pass as a CUDA Graph on first inference,
    then replays it for each iterative decoding step. This eliminates
    kernel launch overhead across 32 iterations × 28 layers.

    Args:
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string.
        max_graphs: Maximum number of per-shape CUDA Graphs kept alive at once.
            Lower this on low-VRAM GPUs; each cached graph pins a private
            activation pool. Default: ``DEFAULT_MAX_GRAPHS``.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
        max_graphs: int = DEFAULT_MAX_GRAPHS,
    ) -> None:
        super().__init__(device=device, model_id=model_id, dtype=dtype)
        self._graph_forward: _CUDAGraphForward | None = None
        self._max_graphs = max_graphs

    def load_model(self) -> None:
        """Load model and install CUDA Graph forward wrapper."""
        super().load_model()
        self._graph_forward = _CUDAGraphForward(
            self._model, max_graphs=self._max_graphs
        )
        self._model.forward = self._graph_forward
        logger.info(
            "FasterRunner ready (CUDA Graph wrapper installed, max_graphs=%d).",
            self._max_graphs,
        )

    def unload_model(self) -> None:
        """Release graphs and unload model."""
        if self._graph_forward is not None:
            self._graph_forward.clear()
            self._graph_forward = None
        super().unload_model()
