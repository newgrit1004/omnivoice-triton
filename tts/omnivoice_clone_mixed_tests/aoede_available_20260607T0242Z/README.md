# Aoede Clone Mixed-Language Tutor Tests

These tests use actual Aoede TTS voice clone reference samples from
`tts/gemini_tts_clone_samples` and compare mixed tutor-style prompts with:

- the language-specific `language` hint
- `language=en`

## Summary

All generated clone-mode outputs worked well.

When using actual TTS voice clone references, the difference between
`language=en` and the language-specific setting is mostly negligible for these
mixed English tutor sentences. Both variants generally preserve intelligibility
and produce usable speech.

The better default is still to use the language-specific setting when possible,
because it tends to maintain the accent and pronunciation more naturally for the
language-specific text embedded in the sentence.

## Recommendation

For mixed tutor prompts such as:

> In Arabic, the phrase كيف حالك؟ is used to ask "How are you?", and you would
> say it when politely greeting someone.

Prefer:

```text
mode=clone
language=<target language>
```

Use `language=en` as a fallback when a language-specific generation is unstable
or sounds degraded.

## Generated Files

### Spanish

- `spanish_usted_tu_clone_aoede_es.wav`
- `spanish_usted_tu_clone_aoede_en.wav`

Text:

```text
The difference between usted and tú is that usted is for formal phrases while tú is for informal phrases
```

### Arabic

- `arabic_kayfa_haluk_clone_aoede_arb.wav`
- `arabic_kayfa_haluk_clone_aoede_en.wav`

Text:

```text
In Arabic, the phrase كيف حالك؟ is used to ask “How are you?”, and you would say it when politely greeting someone.
```

### Urdu

- `urdu_aap_kaise_hain_clone_aoede_ur.wav`
- `urdu_aap_kaise_hain_clone_aoede_en.wav`

Text:

```text
In Urdu, the phrase آپ کیسے ہیں؟ is used to ask “How are you?”, and you would say it when politely greeting someone.
```

## Notes

Chinese, Hindi, and Thai design-mode tests worked well earlier, but matching
per-language Aoede reference WAVs were not present in `tts/gemini_tts_clone_samples`,
so clone-mode Aoede comparisons were only created for Spanish, Arabic, and Urdu.
