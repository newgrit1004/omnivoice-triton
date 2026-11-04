# OmniVoice Clone API Payload Guide

This server does not expose a JSON clone endpoint. Voice cloning uses
`multipart/form-data` against the shared generation endpoint:

```http
POST /generate
Content-Type: multipart/form-data
```

Do not call `POST /v1/audio/speech/clone` unless a compatibility route has been
added. In this repo, that route is documented as unsupported.

## Base URL

Use whatever host/port your API server is running on:

```bash
BASE_URL="http://127.0.0.1:8002"
```

## Direct Clone Request

Use this when every request uploads the reference audio.

```bash
curl -sS -o out.wav \
  -X POST "$BASE_URL/generate" \
  -F mode=clone \
  -F text="This is the text I want spoken in the cloned voice." \
  -F language=en \
  -F ref_audio=@/full/path/to/reference.wav \
  -F ref_text="Transcript of the reference audio, if available."
```

The response body is WAV audio:

```http
200 OK
Content-Type: audio/wav
```

## Required Clone Fields

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `mode` | Yes | string | Must be `clone`. |
| `text` | Yes | string | Text to synthesize in the cloned voice. Blank strings are rejected. |
| `ref_audio` | Required unless `prompt_id` is used | file | Reference speaker audio uploaded as multipart file data. |
| `prompt_id` | Required unless `ref_audio` is used | string | Reusable prompt ID returned by `POST /clone-prompts`. |

For `mode=clone`, send exactly one of:

```text
ref_audio
```

or:

```text
prompt_id
```

Do not send both in the same generation request.

## Optional Clone Fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ref_text` | string | unset | Transcript of `ref_audio`. Recommended, and required when the server runs with `--no-asr` unless `--allow-lazy-asr` is enabled. Cannot be combined with `prompt_id`. |
| `language` | string | auto | OmniVoice language ID or supported language name. Use `GET /languages` to inspect supported values. |
| `instruct` | string | unset | Optional style/control instruction for clone mode, such as `whisper`, `steady`, or `warm conversational voice`. |
| `num_step` | integer | `32` | Decode step override. Must be `> 0`. |
| `guidance_scale` | float | `2.0` | Guidance scale. Must be `>= 0`. |
| `t_shift` | float | `0.1` | Advanced timing shift. Must be `>= 0`. |
| `layer_penalty_factor` | float | `5.0` | Advanced layer penalty. Must be `>= 0`. |
| `position_temperature` | float | `5.0` | Advanced position sampling temperature. Must be `>= 0`. |
| `class_temperature` | float | `0.0` | Advanced class sampling temperature. Must be `>= 0`. |
| `speed` | float | unset | Speech speed override. Must be `> 0` if provided. |
| `duration` | float | unset | Target duration in seconds. Non-positive values are treated as unset. |
| `denoise` | boolean | `true` | Whether denoise behavior is enabled. |
| `preprocess_prompt` | boolean | `true` | Whether to preprocess/clean up the clone prompt. |
| `postprocess_output` | string/boolean | `true` | Legacy postprocess selector. Accepts `true`, `false`, `full`, `light`, or `off`. |
| `postprocess_mode` | string | unset | Explicit postprocess mode. Accepts `full`, `light`, or `off`. Overrides `postprocess_output` when provided. |

## Recommended Direct Clone Payload

For normal use:

```text
mode=clone
text=<target speech text>
language=<language id>
ref_audio=<uploaded wav/mp3/etc file>
ref_text=<reference audio transcript>
num_step=32
guidance_scale=2.0
denoise=true
preprocess_prompt=true
postprocess_mode=full
```

Supplying `ref_text` is the safest production path. It prevents a request from
loading Whisper ASR onto the GPU just to transcribe the reference clip.

For faster/lighter output postprocessing:

```text
mode=clone
text=<target speech text>
language=en
ref_audio=<uploaded reference audio>
ref_text=<reference transcript>
num_step=16
postprocess_mode=light
```

## Reusable Prompt Flow

Use this for repeated requests with the same reference voice. It avoids
rebuilding the clone prompt on every generation request.

### 1. Register Reference Audio

```bash
curl -sS \
  -X POST "$BASE_URL/clone-prompts" \
  -F ref_audio=@/full/path/to/reference.wav \
  -F ref_text="Transcript of the reference audio." \
  -F preprocess_prompt=true
```

Successful response:

```json
{
  "status": "ok",
  "prompt_id": "abc123...",
  "request_id": "e9f1c2d3a4b5",
  "created_at": "2026-05-03T00:00:00Z",
  "duration_ms": 123.45,
  "prompt_audio_tokens": 456,
  "ref_text": "Transcript of the reference audio.",
  "stored_device": "cuda:0"
}
```

### 2. Generate With `prompt_id`

```bash
curl -sS -o out.wav \
  -X POST "$BASE_URL/generate" \
  -F mode=clone \
  -F text="This reuses the registered voice prompt." \
  -F language=en \
  -F prompt_id="abc123..."
```

When using `prompt_id`, do not send `ref_audio` or `ref_text`.

## `/clone-prompts` Payload

```http
POST /clone-prompts
Content-Type: multipart/form-data
```

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `ref_audio` | Yes | file | Reference speaker audio. Empty files are rejected. |
| `ref_text` | No | string | Transcript of the reference audio. |
| `preprocess_prompt` | No | boolean | Defaults to `true`. |

## JavaScript `fetch` Example

```js
const form = new FormData();
form.append("mode", "clone");
form.append("text", "This is the cloned voice output.");
form.append("language", "en");
form.append("ref_text", "Transcript of the reference audio.");
form.append("ref_audio", referenceAudioFile);

const response = await fetch(`${BASE_URL}/generate`, {
  method: "POST",
  body: form,
});

if (!response.ok) {
  throw new Error(await response.text());
}

const wavBlob = await response.blob();
```

Important: do not manually set the `Content-Type` header when using
`FormData` in the browser. The browser will set the multipart boundary.

## Python `requests` Example

```python
import requests

base_url = "http://127.0.0.1:8002"

data = {
    "mode": "clone",
    "text": "This is the cloned voice output.",
    "language": "en",
    "ref_text": "Transcript of the reference audio.",
    "num_step": "32",
    "guidance_scale": "2.0",
    "denoise": "true",
    "preprocess_prompt": "true",
}

with open("/full/path/to/reference.wav", "rb") as audio_file:
    files = {
        "ref_audio": ("reference.wav", audio_file, "audio/wav"),
    }
    response = requests.post(
        f"{base_url}/generate",
        data=data,
        files=files,
        timeout=120,
    )

response.raise_for_status()

with open("out.wav", "wb") as output_file:
    output_file.write(response.content)
```

## Response Headers

Successful `/generate` responses include useful metrics:

| Header | Description |
| --- | --- |
| `X-OmniVoice-Request-Id` | Server request ID. |
| `X-OmniVoice-Latency-Ms` | End-to-end request latency. |
| `X-OmniVoice-Audio-Duration-S` | Generated audio duration. |
| `X-OmniVoice-RTF` | Real-time factor when audio duration is available. |
| `X-OmniVoice-Runner` | Runner implementation. |
| `X-OmniVoice-Device` | Device used by the server. |
| `X-OmniVoice-Prompt-Prepare-Ms` | Time spent preparing clone prompt. |
| `X-OmniVoice-Prompt-Source` | `upload_miss`, `upload_hit`, `cache_disabled`, `prompt_id`, or `none`. |
| `X-OmniVoice-Prompt-Id` | Present when generation used a registered `prompt_id`. |
| `X-OmniVoice-Postprocess-Mode` | Effective postprocess mode: `full`, `light`, or `off`. |

## Validation Rules

Clone requests are rejected when:

| Status | Cause |
| --- | --- |
| `400` | `text` is missing or blank. |
| `400` | `mode=clone` has neither `ref_audio` nor `prompt_id`. |
| `400` | `mode=clone` includes both `ref_audio` and `prompt_id`. |
| `400` | `prompt_id` is combined with `ref_text`. |
| `400` | `ref_audio` is empty. |
| `400` | Numeric fields are invalid, such as `num_step=0` or `speed=-1`. |
| `404` | `prompt_id` is unknown or expired/evicted from the prompt store. |
| `503` | Model, batcher, or registered prompt store is not ready. |

## Common Mistakes

### Sending JSON

This will not work for audio upload:

```http
Content-Type: application/json
```

Use `multipart/form-data`.

### Using the Wrong Clone Route

This repo documents the OpenAI/Chatterbox-style route as unsupported:

```http
POST /v1/audio/speech/clone
```

Use:

```http
POST /generate
```

with:

```text
mode=clone
```

### Combining `prompt_id` With `ref_audio`

Invalid:

```text
mode=clone
text=Hello
prompt_id=abc123
ref_audio=@reference.wav
```

Valid direct upload:

```text
mode=clone
text=Hello
ref_audio=@reference.wav
ref_text=Reference transcript
```

Valid reusable prompt:

```text
mode=clone
text=Hello
prompt_id=abc123
```

### Bad `curl` File Path

If `curl` prints:

```text
curl: (26) Failed to open/read local data from file/application
```

the local file path after `@` is wrong. Prefer an absolute path:

```bash
-F ref_audio=@/full/path/to/reference.wav
```

## Quick Health Check

Before testing clone:

```bash
curl -sS "$BASE_URL/health"
```

To inspect supported languages:

```bash
curl -sS "$BASE_URL/languages"
```

## Minimal Working Checklist

1. Server is running and `/health` returns OK.
2. Request uses `POST /generate`.
3. Request uses `multipart/form-data`.
4. `mode` is `clone`.
5. `text` is present and non-empty.
6. Exactly one of `ref_audio` or `prompt_id` is present.
7. If using `ref_audio`, the file exists and is non-empty.
8. If using `prompt_id`, do not send `ref_text`.
9. Save the response body as `.wav`.
