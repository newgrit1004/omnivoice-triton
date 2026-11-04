"""CPU fake-codec contract checks; not real-tokenizer numerical parity tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omnivoice_triton.models.base_runner import _decode_audio_token_batch


class _FakeAudioTokenizer:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode(self, tokens: torch.Tensor) -> SimpleNamespace:
        self.calls.append(tokens.clone())
        # Encode every token into the output so ordering/length/padding mistakes
        # cannot pass by returning a fixed dummy waveform.
        samples = tokens.to(torch.float32).sum(dim=1).repeat_interleave(3, dim=-1)
        return SimpleNamespace(audio_values=samples.unsqueeze(1))


def _crossfade(chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
    assert sample_rate == 24000
    # Explicitly distinct chunk boundaries, so the fallback's original order is
    # verified. Actual OmniVoice crossfade remains the production callable.
    gap = np.full((1, 2), -1, dtype=np.float32)
    return np.concatenate(
        [
            part
            for index, chunk in enumerate(chunks)
            for part in ([gap, chunk] if index else [chunk])
        ],
        axis=-1,
    )


def _decode(tokenizer, results, size):
    return _decode_audio_token_batch(
        results,
        audio_tokenizer=tokenizer,
        sampling_rate=24000,
        cross_fade_chunks=_crossfade,
        max_batch_size=size,
    )


def test_equal_lengths_share_codec_call_and_preserve_every_sample() -> None:
    results = [torch.full((2, 5), value, dtype=torch.long) for value in range(5)]
    serial, grouped = _FakeAudioTokenizer(), _FakeAudioTokenizer()
    expected = _decode(serial, results, 1)
    actual = _decode(grouped, results, 3)
    assert [tuple(call.shape) for call in serial.calls] == [(1, 2, 5)] * 5
    assert [tuple(call.shape) for call in grouped.calls] == [(3, 2, 5), (2, 2, 5)]
    for reference, waveform in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, waveform)


def test_mixed_lengths_and_dtypes_never_pad_or_reorder() -> None:
    results = [
        torch.full((2, 3), 1, dtype=torch.long),
        torch.full((2, 5), 2, dtype=torch.long),
        torch.full((2, 3), 3, dtype=torch.long),
        torch.full((2, 3), 4, dtype=torch.int32),
        torch.full((3, 3), 5, dtype=torch.long),
    ]
    serial, grouped = _FakeAudioTokenizer(), _FakeAudioTokenizer()
    expected = _decode(serial, results, 1)
    actual = _decode(grouped, results, 8)
    assert [tuple(call.shape) for call in grouped.calls] == [
        (2, 2, 3),
        (1, 2, 5),
        (1, 2, 3),
        (1, 3, 3),
    ]
    assert grouped.calls[2].dtype == torch.int32
    assert [audio.shape[-1] for audio in actual] == [9, 15, 9, 9, 9]
    for reference, waveform in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, waveform)


def test_chunk_lists_keep_serial_decoder_and_crossfade_order() -> None:
    first = torch.full((2, 3), 1, dtype=torch.long)
    last = torch.full((2, 3), 3, dtype=torch.long)
    chunks = [torch.full((2, 2), value, dtype=torch.long) for value in (4, 7)]
    results = [first, chunks, last]
    serial, grouped = _FakeAudioTokenizer(), _FakeAudioTokenizer()
    expected = _decode(serial, results, 1)
    actual = _decode(grouped, results, 8)
    assert [tuple(call.shape) for call in grouped.calls] == [
        (1, 2, 2),
        (1, 2, 2),
        (2, 2, 3),
    ]
    for reference, waveform in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, waveform)


def test_invalid_codec_batch_dimension_is_rejected() -> None:
    class BrokenTokenizer(_FakeAudioTokenizer):
        def decode(self, tokens):
            return SimpleNamespace(audio_values=torch.zeros((1, 1, 10)))

    with pytest.raises(RuntimeError, match="invalid batch dimension"):
        _decode(BrokenTokenizer(), [torch.zeros((2, 3))] * 2, 8)


def test_empty_batch_does_not_invoke_decoder() -> None:
    tokenizer = _FakeAudioTokenizer()
    assert _decode(tokenizer, [], 8) == []
    assert tokenizer.calls == []
