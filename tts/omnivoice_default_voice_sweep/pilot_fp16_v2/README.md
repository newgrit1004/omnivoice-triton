# OmniVoice Default Voice Sweep

This run compares OmniVoice default/design voices against optional Gemini clone baselines.

## Config

- runner: `hybrid`
- dtype: `fp16`
- profile_set: `pilot`
- num_steps: `[16]`
- languages: `['en', 'es', 'fr', 'de', 'ar', 'ur']`
- include_gemini_clones: `True`

## Files

- `manifest.json`: all generated sample metadata
- `instruct_catalog.json`: official and exploratory voice-design instructions
- `review.html`: audio-player review sheet with manual rating columns

## Notes

- Official OmniVoice voice-design tags include gender, age, pitch, whisper, English accent, and Chinese dialect.
- Deep voice experiments use `low pitch` and `very low pitch`.
- Exploratory profiles keep tutor/persona words in metadata, because OmniVoice 0.1.4 rejects unsupported prose inside `instruct`.
