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


class CudaGraphCaptureError(RuntimeError):
    """A CUDA Graph capture failed.

    The process is not safe to keep serving afterwards: in production a failed capture
    was followed seconds later by "device-side assert triggered" on every request.
    """


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
    """

    def __init__(self, model: Any, *, max_graphs: int = 32) -> None:
        self._model = model
        self._original_forward = model.forward
        self.max_graphs = max(0, int(max_graphs))
        self._graphs: OrderedDict[tuple[int, ...], dict] = OrderedDict()

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
        # "thread_local": only this thread's unsafe calls can invalidate the capture.
        # With the default "global" mode, ordinary CUDA work in any other thread (a
        # voice prompt being encoded for /clone-prompts) aborted it. The API server also
        # keeps all of its CUDA work on the batcher thread that captures; this guards
        # against anything outside it.
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                static_output = self._original_forward(
                    static_input_ids,
                    static_audio_mask,
                    **kwargs,
                )
        except Exception as exc:
            raise CudaGraphCaptureError(
                f"CUDA Graph capture failed for shape {key}: {exc}"
            ) from exc

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
        self._graphs.move_to_end(key)
        while self.max_graphs > 0 and len(self._graphs) > self.max_graphs:
            evicted_key, _ = self._graphs.popitem(last=False)
            logger.info("Evicted CUDA Graph for shape %s", evicted_key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

        if self.max_graphs <= 0:
            return self._original_forward(
                input_ids,
                audio_mask,
                labels,
                attention_mask,
                document_ids,
                position_ids,
            )

        key = self._get_shape_key(input_ids)

        if key not in self._graphs:
            entry = self._capture(
                input_ids,
                audio_mask,
                attention_mask,
                document_ids,
                position_ids,
            )
        else:
            entry = self._graphs[key]
            self._graphs.move_to_end(key)

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
        """Release all captured graphs."""
        self._graphs.clear()

    def prewarm_shape(self, shape: tuple[int, int, int]) -> bool:
        """Capture a CUDA Graph for a shape before live traffic arrives."""
        if self.max_graphs <= 0:
            return False
        if shape in self._graphs:
            return False

        batch_size, num_audio_codebook, seq_len = shape
        device = getattr(self._model, "device", "cuda")
        pad_id = int(getattr(self._model.config, "audio_mask_id", 0))
        input_ids = torch.full(
            shape,
            pad_id,
            dtype=torch.long,
            device=device,
        )
        audio_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=device,
        )
        audio_mask[:, seq_len // 2 :] = True
        attention_mask = torch.tril(
            torch.ones(
                (batch_size, 1, seq_len, seq_len),
                dtype=torch.bool,
                device=device,
            )
        )

        expected_codebooks = int(
            getattr(self._model.config, "num_audio_codebook", 0)
        )
        if num_audio_codebook != expected_codebooks:
            raise ValueError(
                "CUDA Graph prewarm shape codebook dimension does not match "
                "model config."
            )

        self._capture(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
        )
        return True

    def get_metrics(self) -> dict[str, Any]:
        """Return summary metrics for captured CUDA Graph shapes."""
        return {
            "capture_count": len(self._graphs),
            "max_graphs": self.max_graphs,
            "shapes": [list(shape) for shape in sorted(self._graphs)],
        }


class FasterRunner(BaseRunner):
    """BaseRunner with CUDA Graph optimization for the forward pass.

    Captures the OmniVoice forward pass as a CUDA Graph on first inference,
    then replays it for each iterative decoding step. This eliminates
    kernel launch overhead across 32 iterations × 28 layers.

    Args:
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
        decode_postprocess_workers: int = 0,
        max_cuda_graphs: int = 32,
    ) -> None:
        super().__init__(
            device=device,
            model_id=model_id,
            dtype=dtype,
            decode_postprocess_workers=decode_postprocess_workers,
        )
        self._graph_forward: _CUDAGraphForward | None = None
        self.max_cuda_graphs = max(0, int(max_cuda_graphs))

    def load_model(self) -> None:
        """Load model and install CUDA Graph forward wrapper."""
        super().load_model()
        self._graph_forward = _CUDAGraphForward(
            self._model,
            max_graphs=self.max_cuda_graphs,
        )
        self._model.forward = self._graph_forward
        logger.info(
            "FasterRunner ready (CUDA Graph wrapper installed, max_graphs=%d).",
            self.max_cuda_graphs,
        )

    def unload_model(self) -> None:
        """Release graphs and unload model."""
        if self._graph_forward is not None:
            self._graph_forward.clear()
            self._graph_forward = None
        super().unload_model()

    def get_cuda_graph_metrics(self) -> dict[str, Any]:
        """Return current CUDA Graph capture metrics."""
        if self._graph_forward is None:
            return {"capture_count": 0, "shapes": []}
        return self._graph_forward.get_metrics()

    def prewarm_cuda_graph_shapes(
        self,
        shapes: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    ) -> dict[str, Any]:
        """Pre-capture CUDA Graphs for known hot-path shapes."""
        if self._graph_forward is None:
            return {"capture_count": 0, "shapes": [], "warmed_shapes": []}

        warmed_shapes: list[list[int]] = []
        for shape in shapes:
            if len(shape) != 3:
                raise ValueError(f"Expected 3D CUDA Graph shape, got {shape!r}")
            normalized = tuple(int(dim) for dim in shape)
            if self._graph_forward.prewarm_shape(normalized):
                warmed_shapes.append(list(normalized))

        metrics = self._graph_forward.get_metrics()
        metrics["warmed_shapes"] = warmed_shapes
        return metrics
