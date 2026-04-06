# Audio Samples

Pre-generated audio samples demonstrating each OmniVoice runner mode.

## Directory Structure

```
assets/audio_samples/
├── base/               # BaseRunner samples
│   ├── ko_00.wav
│   ├── ko_01.wav
│   ├── ko_02.wav
│   ├── en_00.wav
│   ├── en_01.wav
│   ├── en_02.wav
│   ├── zh_00.wav
│   ├── zh_01.wav
│   ├── zh_02.wav
│   ├── voice_design_female_young_cheerful.wav
│   └── voice_design_male_adult_calm.wav
├── triton/             # TritonRunner samples (same structure)
├── faster/             # FasterRunner samples (same structure)
├── hybrid/             # TritonFasterRunner samples (same structure)
└── metadata.json       # Generation metadata and timing info
```

## Sample Types

| Type | Description |
|------|-------------|
| `ko_*.wav` | Korean text samples |
| `en_*.wav` | English text samples |
| `zh_*.wav` | Chinese text samples |
| `voice_design_*.wav` | Voice design samples with style instructions |

## metadata.json Schema

```json
{
  "generated_at": "<ISO 8601 timestamp>",
  "hardware": {
    "gpu": "<GPU name>",
    "cuda_version": "<version>",
    "vram_gb": 0.0
  },
  "modes": ["base", "triton", "faster", "hybrid"],
  "samples": [
    {
      "mode": "base",
      "type": "generate",
      "language": "en",
      "text": "Hello, nice to meet you.",
      "file": "base/en_00.wav",
      "sample_rate": 24000,
      "generation_time_s": 0.512
    }
  ]
}
```

## Regenerating Samples

```bash
# All modes
make generate-samples

# Specific modes only
uv run python scripts/generate_samples.py --modes base triton

# Custom output directory
uv run python scripts/generate_samples.py --output-dir /tmp/my_samples
```

All audio is saved as 32-bit float WAV at 24 kHz.
