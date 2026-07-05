"""Regression tests for the bounded CUDA Graph cache (issue #8).

An unbounded per-shape graph cache leaked VRAM when inputs had many distinct
sequence lengths. These tests verify the cache is capped and that overflow
shapes fall back to the eager forward instead of capturing new graphs. They
mock graph capture so they run on CPU (no GPU required).
"""

from unittest.mock import MagicMock

import torch

from omnivoice_triton.models.faster_runner import _CUDAGraphForward


def _make_wrapper(max_graphs):
    """Build a _CUDAGraphForward whose capture is mocked (no real CUDA graph)."""
    model = MagicMock()
    model.training = False
    wrapper = _CUDAGraphForward(model, max_graphs=max_graphs)

    # Count eager-forward invocations; return an object with a stable shape.
    wrapper.eager_calls = 0

    def fake_original(input_ids, audio_mask, *args, **kwargs):
        wrapper.eager_calls += 1
        return MagicMock(name="eager_output")

    wrapper._original_forward = fake_original

    # Mocked capture: store real CPU static buffers so replay's copy_ works.
    def fake_capture(input_ids, audio_mask, attention_mask=None,
                     document_ids=None, position_ids=None):
        key = wrapper._get_shape_key(input_ids)
        entry = {
            "graph": MagicMock(),
            "static_input_ids": input_ids.clone(),
            "static_audio_mask": audio_mask.clone(),
            "static_attn_mask": None,
            "static_doc_ids": None,
            "static_pos_ids": None,
            "static_output": MagicMock(name=f"replay_output_{key}"),
        }
        wrapper._graphs[key] = entry
        return entry

    wrapper._capture = fake_capture
    return wrapper


def _inputs(seq_len):
    return torch.zeros(2, 8, seq_len, dtype=torch.long), torch.zeros(2, 8, seq_len)


def test_cache_is_bounded_by_max_graphs():
    w = _make_wrapper(max_graphs=4)
    for seq in range(10, 30):  # 20 distinct shapes
        w(*_inputs(seq))
    assert len(w._graphs) == 4, "graph cache must not exceed max_graphs"


def test_overflow_shapes_use_eager_fallback():
    w = _make_wrapper(max_graphs=3)
    for seq in (10, 11, 12):        # fill the cache
        w(*_inputs(seq))
    assert len(w._graphs) == 3
    assert w.eager_calls == 0

    for seq in (13, 14, 15):        # overflow -> eager, no new graphs
        w(*_inputs(seq))
    assert len(w._graphs) == 3, "overflow must not capture new graphs"
    assert w.eager_calls == 3, "overflow shapes must run the eager forward"


def test_cached_shape_replays_without_recapture():
    w = _make_wrapper(max_graphs=2)
    a = _inputs(10)
    out1 = w(*a)
    graph1 = w._graphs[(2, 8, 10)]["graph"]
    out2 = w(*a)                    # same shape -> replay, not recapture
    assert w._graphs[(2, 8, 10)]["graph"] is graph1
    assert graph1.replay.call_count == 2
    assert out1 is out2             # returns the same static output buffer


def test_clear_releases_all_graphs():
    w = _make_wrapper(max_graphs=4)
    for seq in (10, 11, 12, 13):
        w(*_inputs(seq))
    assert len(w._graphs) == 4
    w.clear()
    assert len(w._graphs) == 0
    assert len(w._fallback_warned) == 0


def test_invalid_max_graphs_rejected():
    import pytest

    with pytest.raises(ValueError):
        _CUDAGraphForward(MagicMock(), max_graphs=0)
