# OmniVoice Server API Calls

This worker is an OmniVoice HTTP server. It does not expose Chatterbox or
OpenAI-style TTS routes.

Use one of these base URLs:

```bash
BASE_URL=http://127.0.0.1:8000
# Or the public Vast.ai URL reported by GET /worker:
# BASE_URL=http://74.48.140.178:43531
```

## Supported Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Full server, model, GPU, batching, and worker health. |
| `GET` | `/healthz` | Compact readiness probe for Lingua worker checks. |
| `GET` | `/languages` | Supported OmniVoice language IDs and names. |
| `GET` | `/worker` | Worker callback status, assignability, capacity, and public URL. |
| `GET` | `/worker/state` | Alias of `/worker`. |
| `POST` | `/drain` | Marks the worker as draining. Requires `X-Worker-Token`. |
| `POST` | `/clone-prompts` | Registers reusable clone prompt audio and returns a `prompt_id`. |
| `POST` | `/cache/warm` | Prepares clone prompt cache entries. |
| `GET` | `/cache/status` | Reports whether a worker-local cache key or `prompt_id` is ready. |
| `POST` | `/generate` | Generates speech and returns inline `audio/wav`. |

FastAPI discovery endpoints are also available:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/openapi.json` | Machine-readable API schema. |
| `GET` | `/docs` | Swagger UI. |
| `GET` | `/redoc` | ReDoc UI. |

## Health

```bash
curl -sS "$BASE_URL/health"
```

Expected success: `200 OK` JSON with `"status": "ok"`.

## Worker State

```bash
curl -sS "$BASE_URL/worker"
```

Expected success: `200 OK` JSON similar to:

```json
{
  "worker_type": "tts",
  "worker_id": "tts-36043713",
  "base_url": "http://74.48.140.178:43531",
  "callback_enabled": true,
  "status": "healthy",
  "active_requests": 0,
  "queue_depth": 0,
  "capacity": 1,
  "assignable": true,
  "draining": false
}
```

## Languages

```bash
curl -sS "$BASE_URL/languages"
```

Use the returned language IDs when possible. For Arabic, use a specific
OmniVoice language ID such as `arb` for Standard Arabic. Plain `ar` or
`Arabic` may fall back to language-agnostic mode.

## Generate Speech

`POST /generate` expects `multipart/form-data` and returns `audio/wav`.

Required fields:

| Field | Required | Notes |
| --- | --- | --- |
| `mode` | Yes | One of `auto`, `design`, or `clone`. |
| `text` | Yes | Text to synthesize. |

Common optional fields:

| Field | Notes |
| --- | --- |
| `language` | OmniVoice language ID or supported language name. |
| `instruct` | Voice design instruction. Used with `mode=design`; optional in `mode=clone`. |
| `prompt_id` | Reusable clone prompt ID from `/clone-prompts`. Used with `mode=clone`. |
| `ref_audio` | Uploaded reference audio file. Used with `mode=clone`. |
| `ref_text` | Optional transcript for `ref_audio`; required when `--no-asr` is used unless `--allow-lazy-asr` is enabled. |
| `num_step` | Decode step override. |
| `guidance_scale` | Guidance scale override. |
| `t_shift` | Advanced generation timing shift override. |
| `layer_penalty_factor` | Advanced layer penalty override. |
| `position_temperature` | Advanced position sampling temperature override. |
| `class_temperature` | Advanced class sampling temperature override. |
| `speed` | Speech speed override. |
| `duration` | Target duration override. |
| `denoise` | Boolean, default `true`. |
| `preprocess_prompt` | Boolean, default `true`. |
| `postprocess_output` | Boolean-like string, default `true`. |
| `postprocess_mode` | Optional postprocess mode override. |
| `insert_pauses` | Boolean, default `false`. Synthesize sentences separately and insert 300 ms of silence between them after post-processing. |

### Auto Mode

Use this for normal text-only TTS.

```bash
curl -sS -o out.wav \
  -X POST "$BASE_URL/generate" \
  -F mode=auto \
  -F text="Marhaban" \
  -F language=arb
```

`mode=auto` does not accept `instruct`, `prompt_id`, `ref_audio`, or `ref_text`.

### Design Mode

Use this for text plus a voice/style instruction.

```bash
curl -sS -o out.wav \
  -X POST "$BASE_URL/generate" \
  -F mode=design \
  -F text="Good evening. How is your night going?" \
  -F language=en \
  -F instruct="sweet, warm, conversational voice"
```

`mode=design` does not accept `prompt_id`, `ref_audio`, or `ref_text`.

### Clone Mode With Uploaded Reference Audio

Use this when each request includes a reference audio file.

```bash
curl -sS -o out.wav \
  -X POST "$BASE_URL/generate" \
  -F mode=clone \
  -F text="This is a cloned voice sample." \
  -F language=en \
  -F ref_audio=@clone.wav \
  -F ref_text="Reference transcript, if available."
```

`mode=clone` requires exactly one of `prompt_id` or `ref_audio`.
Supplying `ref_text` is recommended so the worker does not need to load Whisper
ASR onto the GPU.

### Clone Mode With Reusable Prompt ID

First register the reference audio:

```bash
curl -sS \
  -X POST "$BASE_URL/clone-prompts" \
  -F ref_audio=@clone.wav \
  -F ref_text="Reference transcript, if available."
```

The response includes `prompt_id`. Use it in later generation calls:

```bash
curl -sS -o out.wav \
  -X POST "$BASE_URL/generate" \
  -F mode=clone \
  -F text="This reuses the registered voice prompt." \
  -F language=en \
  -F prompt_id=REPLACE_WITH_PROMPT_ID
```

## Drain Worker

Drain mode makes the worker stop accepting new generation requests while
allowing in-flight work to finish.

```bash
curl -sS \
  -X POST "$BASE_URL/drain" \
  -H "X-Worker-Token: $LINGUA_WORKER_TOKEN"
```

Expected success: `200 OK` JSON with `"draining": true` and
`"assignable": false`.

## Routes That Return 404 On This Server

These are not implemented by the OmniVoice worker:

| Method | Path | Why it fails |
| --- | --- | --- |
| `GET` | `/status` | Use `GET /health` or `GET /worker`. |
| `POST` | `/v1/tts` | Chatterbox-style route; use `POST /generate`. |
| `POST` | `/v1/audio/speech` | OpenAI-style route; use `POST /generate`. |
| `POST` | `/v1/audio/speech/clone` | Chatterbox/OpenAI-style clone route; use `POST /generate` with `mode=clone`. |

If the backend or mobile app is configured for Chatterbox default synthesis, it
must either map those requests to `POST /generate` or the worker must add
compatibility routes for the Chatterbox paths.

## Worker Callback Behavior

The worker registers itself with the Lingua control plane when these environment
variables are present:

```bash
LINGUA_CONTROL_PLANE_BASE_URL=http://18.205.211.142:8000
LINGUA_WORKER_REGISTER_URL=http://18.205.211.142:8000/api/runtime/workers/register
LINGUA_WORKER_HEARTBEAT_URL=http://18.205.211.142:8000/api/runtime/workers/heartbeat
LINGUA_WORKER_TOKEN=...
LINGUA_WORKER_PUBLIC_BASE_URL=http://74.48.140.178:43531
```

Those register and heartbeat calls are outgoing callbacks from the worker to the
control plane. They are not inbound routes served by this OmniVoice worker.

### Optional sentence pauses

Add `-F 'insert_pauses=true'` to any `/generate` request, for example:

```bash
curl -X POST "$BASE_URL/generate" \
  -F 'mode=auto' \
  -F 'text=Hello there. How are you today?' \
  -F 'postprocess_mode=light' \
  -F 'insert_pauses=true' \
  --output speech.wav
```

This works in `auto`, `design`, and `clone` modes with `full`, `light`, or `off`
post-processing. The default remains unchanged. Single sentences receive no added
silence, and no gap is added before the first or after the last sentence. Existing
model-generated silence is preserved, so the total boundary pause can exceed 300 ms.

Sentence splitting uses punctuation (`.`, `!`, `?`, `。`, `！`, `？`), preserving
common English abbreviations, initials, decimals, and closing quotes. It is a
heuristic, not a language-specific sentence parser. Each sentence is a separate
synthesis using the same language, instruction, speed, and clone prompt. This can
increase latency and change prosody or voice consistency, especially without a
clone prompt. Use `speed` to adjust pacing; an explicit `duration` with multiple
sentences and `insert_pauses=true` returns HTTP 400.

Response headers include `X-OmniVoice-Insert-Pauses` and
`X-OmniVoice-Sentence-Count`. WAV duration and saved files include inserted silence.
For multiple sentences, batch timing, request counts, token totals, and generation
stage timings sum across participating batches (which may include other requests);
sequence length and VRAM report maxima. Batch lane reports the last sentence's
lane. GPU sample metrics are omitted because separate sampling windows cannot be
accurately merged.
