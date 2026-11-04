# OmniVoice Voice Sweep Notes

This document tracks the Lingua experiment for replacing Gemini TTS clone
reference voices with OmniVoice default/design voices.

## Valid Instruct Tags

OmniVoice 0.1.4 does not accept open-ended prose in `instruct`. It validates a
controlled set of tags from the upstream Gradio demo and
`omnivoice.utils.voice_design`.

English tags:

- gender: `male`, `female`
- age: `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
- pitch: `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`,
  `very high pitch`
- style: `whisper`
- English accent: `american accent`, `british accent`, `australian accent`,
  `chinese accent`, `canadian accent`, `indian accent`, `korean accent`,
  `portuguese accent`, `russian accent`, `japanese accent`

Chinese dialect tags:

- `河南话`, `陕西话`, `四川话`, `贵州话`, `云南话`, `桂林话`, `济南话`,
  `石家庄话`, `甘肃话`, `宁夏话`, `青岛话`, `东北话`

Persona words such as "warm", "trusting", "clear", or "tutor voice" are kept
as metadata labels only. Passing them inside `instruct` causes validation
errors.

## Generated Runs

Completed runs live under `tts/omnivoice_default_voice_sweep/`.

- `official_params_fp16`: 339 design samples, 0 errors. Covers the official
  curated tags, English accents, and Chinese dialects.
- `pilot_fp16_v2`: 212 samples. Includes 180 curated design samples and 32
  clone samples. Clone generation hit OOMs after many graph shapes, so this is
  useful but not the clean clone baseline.
- `clone_baselines_fp16_graph4`: 57 clone samples, 0 errors. Clean Gemini clone
  baseline rerun with `--max-cuda-graphs 4`.

## Coverage Gap

The first official sweep had broad tag coverage, but it did not cover the full
gender x age x pitch matrix. It included examples such as:

- `female, young adult, high pitch`
- `male, middle-aged, low pitch`
- `male, middle-aged, very low pitch`
- `female, elderly, low pitch`
- `child, high pitch`

Missing examples included many combinations such as:

- `female, child, very low pitch`
- `male, teenager, very high pitch`
- `female, elderly, high pitch`
- `male, young adult, very low pitch`
- `female, middle-aged, very high pitch`

To cover this gap, `scripts/generate_default_voice_sweep.py` now has a
`matrix` profile set:

- 2 genders x 5 ages x 5 pitches = 50 profiles
- 2 genders x 5 ages at `moderate pitch, whisper` = 10 profiles
- total: 60 profile combinations

Recommended matrix run:

```bash
.venv/bin/python scripts/generate_default_voice_sweep.py \
  --languages en,es,fr,de,ar,ur,zh \
  --profile-set matrix \
  --text-case baseline \
  --run-id age_pitch_matrix_fp16 \
  --dtype fp16 \
  --device cuda \
  --num-step 16 \
  --max-cuda-graphs 4
```

This creates 420 samples: 60 profiles x 7 languages x 1 baseline text case.
Use the baseline-only matrix for voice timbre selection first. After narrowing
to a few good candidates, rerun those profiles on pronunciation and mixed tutor
texts.

## Clean Clone Baseline Command

The clone baseline should use a smaller graph cache to avoid VRAM pressure from
many large reference-audio graph shapes:

```bash
.venv/bin/python scripts/generate_default_voice_sweep.py \
  --languages en,es,fr,de,ar,ur \
  --profile-set pilot \
  --include-gemini-clones \
  --skip-design \
  --gemini-voices Aoede,Despina,Charon,Fenrir \
  --run-id clone_baselines_fp16_graph4 \
  --dtype fp16 \
  --device cuda \
  --num-step 16 \
  --max-cuda-graphs 4
```
