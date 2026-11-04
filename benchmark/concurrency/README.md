# Complete-audio optimization experiments

**Measured GPU: NVIDIA GeForce RTX 3090, physical VRAM 24,576 MiB / PyTorch-visible VRAM 25,296,044,032 bytes.** Driver 580.65.06, CUDA 12.8. Experiment date: November 3, 2026, America/Chicago.

- [Measured results and limits](../../docs/concurrency-speedups-2026-11-03.md)
- [Saved before/after complete WAVs](samples/rtx3090-2026-11-03/README.md)
- [TTS benchmark protocol](bench/README.md)
- [Offline Omnilingual ASR judges](asr/README.md)

The 32-graph profile was evaluated only on this 24 GB RTX 3090. The experimental codec batching flag defaults to 1, preserving serial decoding. ASR runs after TTS timing on the GPU. No paid transcription API is used.
