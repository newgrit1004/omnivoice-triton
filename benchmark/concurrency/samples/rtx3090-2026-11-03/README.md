# Saved before/after complete WAVs

**Measured GPU: NVIDIA GeForce RTX 3090, physical VRAM 24,576 MiB / PyTorch-visible VRAM 25,296,044,032 bytes.** Driver 580.65.06, CUDA 12.8. TTS date: November 3, 2026, America/Chicago.

These 15 sequential corpus pairs contain byte-identical WAVs before and after the async/cache-32 profile. Three additional warmup pairs also matched in the full experiment. Startup seed `20261103` was applied and checked before model import; no per-request seed or concurrent determinism is claimed. Identical audio proves preservation for these fixtures. Existing word and ASR issues remain for review.

| Case | Before | After |
| --- | --- | --- |
| single-corpus-0000-en-short | [WAV](before/single-corpus-0000-en-short.wav) | [WAV](after/single-corpus-0000-en-short.wav) |
| single-corpus-0001-en-medium | [WAV](before/single-corpus-0001-en-medium.wav) | [WAV](after/single-corpus-0001-en-medium.wav) |
| single-corpus-0002-en-long | [WAV](before/single-corpus-0002-en-long.wav) | [WAV](after/single-corpus-0002-en-long.wav) |
| single-corpus-0003-es-short | [WAV](before/single-corpus-0003-es-short.wav) | [WAV](after/single-corpus-0003-es-short.wav) |
| single-corpus-0004-es-medium | [WAV](before/single-corpus-0004-es-medium.wav) | [WAV](after/single-corpus-0004-es-medium.wav) |
| single-corpus-0005-es-long | [WAV](before/single-corpus-0005-es-long.wav) | [WAV](after/single-corpus-0005-es-long.wav) |
| single-corpus-0006-ar-short | [WAV](before/single-corpus-0006-ar-short.wav) | [WAV](after/single-corpus-0006-ar-short.wav) |
| single-corpus-0007-ar-medium | [WAV](before/single-corpus-0007-ar-medium.wav) | [WAV](after/single-corpus-0007-ar-medium.wav) |
| single-corpus-0008-ar-long | [WAV](before/single-corpus-0008-ar-long.wav) | [WAV](after/single-corpus-0008-ar-long.wav) |
| single-corpus-0009-en-critical-0 | [WAV](before/single-corpus-0009-en-critical-0.wav) | [WAV](after/single-corpus-0009-en-critical-0.wav) |
| single-corpus-0010-en-critical-1 | [WAV](before/single-corpus-0010-en-critical-1.wav) | [WAV](after/single-corpus-0010-en-critical-1.wav) |
| single-corpus-0011-es-critical-2 | [WAV](before/single-corpus-0011-es-critical-2.wav) | [WAV](after/single-corpus-0011-es-critical-2.wav) |
| single-corpus-0012-es-critical-3 | [WAV](before/single-corpus-0012-es-critical-3.wav) | [WAV](after/single-corpus-0012-es-critical-3.wav) |
| single-corpus-0013-ar-critical-4 | [WAV](before/single-corpus-0013-ar-critical-4.wav) | [WAV](after/single-corpus-0013-ar-critical-4.wav) |
| single-corpus-0014-ar-critical-5 | [WAV](before/single-corpus-0014-ar-critical-5.wav) | [WAV](after/single-corpus-0014-ar-critical-5.wav) |

Full texts, hashes, profile and code/checkpoint revisions are in [manifest.json](manifest.json).
