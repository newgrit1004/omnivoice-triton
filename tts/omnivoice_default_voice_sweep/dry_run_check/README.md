# OmniVoice Default Voice Sweep

This run compares OmniVoice default/design voices against optional Gemini clone baselines.

## Config

- runner: `hybrid`
- dtype: `fp16`
- profile_set: `pilot`
- num_steps: `[16]`
- languages: `['en', 'es']`
- include_gemini_clones: `False`

## Files

- `manifest.json`: all generated sample metadata
- `instruct_catalog.json`: official and exploratory voice-design instructions
- `review.html`: audio-player review sheet with manual rating columns

## Notes

- Official OmniVoice voice-design tags include gender, age, pitch, whisper, English accent, and Chinese dialect.
- Deep voice experiments use `low pitch` and `very low pitch`.
- Exploratory profiles add free-form tutor/persona words around official tags.
