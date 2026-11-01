# API Server

This document explains the FastAPI server implemented in
[src/omnivoice_triton/cli/api_server.py](/workspace/omnivoice-triton/src/omnivoice_triton/cli/api_server.py).

The server wraps this repo's runner API and exposes a simple HTTP interface
for OmniVoice generation through the optimized `omnivoice-triton` backends.

It is intended for:

- local testing
- simple app integration
- Vast.ai deployment
- measuring latency and real-time factor
- comparing `base`, `triton`, `faster`, and `hybrid`

It now supports **in-process micro-batching** for concurrent requests.

That means the server no longer runs every request under a single FIFO
generation lock. Instead it:

- prepares each request independently
- caches prepared clone prompts in memory
- queues compatible requests briefly
- merges them into one batched `model.generate(...)` call

It is still **one process with one in-flight merged batch at a time**, so this
is a throughput improvement rather than a full multi-worker serving system.

## Lingua Worker Callback

For Vast.ai autoscaling, the API server can self-register as a Lingua
`worker_type: "tts"` worker. Callback behavior is enabled only when the worker
token and control-plane URL are present in the environment.

Required environment:

- `LINGUA_CONTROL_PLANE_BASE_URL`
- `LINGUA_WORKER_TOKEN`
- `LINGUA_WORKER_TYPE=tts`
- `LINGUA_WORKER_DEFAULT_CAPACITY=1`

Optional overrides:

- `LINGUA_WORKER_REGISTER_URL`
- `LINGUA_WORKER_HEARTBEAT_URL`
- `WORKER_ID`
- `LINGUA_WORKER_PUBLIC_BASE_URL`
- `LINGUA_WORKER_PUBLIC_IP`
- `LINGUA_WORKER_PUBLIC_PORT`
- `LINGUA_WORKER_REGION`
- `LINGUA_WORKER_GPU_TYPE`
- `OMNIVOICE_MODEL_ID`
- `OMNIVOICE_MAX_CONCURRENT`
- `LINGUA_WORKER_HEARTBEAT_INTERVAL_SECONDS`
- `LINGUA_WORKER_DRAIN_TIMEOUT_SECONDS`

If `LINGUA_WORKER_PUBLIC_BASE_URL` is not set, the server attempts to build a
public URL from Vast-style `PUBLIC_IPADDR` and `VAST_TCP_PORT_<port>` variables.
Registration uses `X-Worker-Token: $LINGUA_WORKER_TOKEN`, retries until
successful, and then sends periodic heartbeats with active request and drain
metadata.

Heartbeat intervals are clamped to the Lingua-safe `15` to `30` second window.
If the public Vast URL is not available at startup, registration waits and
keeps retrying URL discovery instead of advertising a local or container-only
address.

Drain mode is available through:

```bash
curl -X POST \
  -H "X-Worker-Token: $LINGUA_WORKER_TOKEN" \
  http://127.0.0.1:8002/drain
```

While draining, new `/generate`, `/clone-prompts`, and `/cache/warm` work
returns `503` with `{"ok": false, "error": "worker_draining"}` and existing
in-flight requests are allowed to finish. Shutdown sends a final `draining`
heartbeat and waits up to `LINGUA_WORKER_DRAIN_TIMEOUT_SECONDS` for in-flight
work before unloading. `/health`, `/healthz`, and `/worker` expose the current
worker status, active request count, assignability, callback config, and public
base URL.

If the CUDA context fails (a device-side assert, an illegal memory access, or a
failed CUDA Graph capture; out-of-memory errors do not count), the worker stops
taking work for good: `/health` and `/healthz` return `503`, the worker status
becomes `dead` (sent to Lingua in an immediate heartbeat), and new work returns
`503` with `{"ok": false, "error": "cuda_fault", "detail": "..."}`. Restart the
process to recover; the failed context cannot be reused.

All CUDA work runs on the batcher thread: generation batches, and voice prompt
encoding for `/clone-prompts`, `/cache/warm`, and uploaded `ref_audio`, which
runs between batches. Encoding a prompt from a request thread while a CUDA Graph
was being captured used to abort the capture and break the process.

## Overview

The server is a thin HTTP layer around:

- `omnivoice_triton.create_runner(...)`
- `runner.load_model()`
- `runner.model.generate(...)`
- `runner.model.create_voice_clone_prompt(...)`

It supports three generation modes:

- `auto`: text-only TTS
- `design`: text plus optional `instruct`
- `clone`: text plus uploaded reference audio, with optional `ref_text`

Successful `/generate` responses return inline `audio/wav`.

## Entry Point

The CLI entrypoint is:

```bash
omnivoice-api
```

It is registered in [pyproject.toml](/workspace/omnivoice-triton/pyproject.toml).

You can also run it as a module:

```bash
cd /workspace/omnivoice-triton
.venv/bin/python -m omnivoice_triton.cli.api_server --runner hybrid --device cuda --ip 0.0.0.0 --port 8002
```

## Startup Behavior

At startup, `create_app(...)`:

1. picks the requested device or auto-detects one
2. creates the requested runner via `create_runner(...)`
3. loads the OmniVoice model into that runner
4. optionally loads ASR at startup unless `--no-asr` is used
5. starts the in-process generation batcher
6. stores the runner and runtime settings in `app.state`

Shutdown unloads the runner and frees GPU memory.

## Runner Modes

Unlike the upstream `omnivoice-demo` Gradio app, this server can run the
optimized backends from this repo.

Available runners:

- `base`: standard PyTorch OmniVoice
- `triton`: Triton kernel fusion
- `faster`: CUDA Graph wrapping
- `hybrid`: Triton + CUDA Graph

Default:

- `hybrid`

Practical recommendation:

- use `hybrid` on CUDA unless you are debugging correctness
- use `base` if you need a non-CUDA fallback

Important device constraint:

- `triton`, `faster`, and `hybrid` require CUDA
- `base` is the only safe fallback for non-CUDA devices

## CLI Options

Main runtime flags:

- `--model`: checkpoint path or Hugging Face repo ID
- `--runner`: `base|triton|faster|hybrid`
- `--device`: explicit device such as `cuda`, `cuda:0`, `cpu`, or `mps`
- `--dtype`: `bf16|fp16|fp32`
- `--ip`: bind address
- `--port`: bind port
- `--root-path`: reverse-proxy root path
- `--no-asr`: skip loading Whisper ASR at startup and require `ref_text`
  unless `--allow-lazy-asr` is also set
- `--allow-lazy-asr`: allow clone requests without `ref_text` to load Whisper
  ASR on demand
- `--save-dir`: persist a copy of each generated WAV
- `--sage-attention`: enable SageAttention on `triton` or `hybrid`
- `--batch-collect-ms`: micro-batch collection window
- `--max-batch-requests`: max requests per merged batch
- `--max-pending-requests`: prepared requests waiting behind the executing batch
  (default 64)
- `--max-pending-gpu-tasks`: reference/cache tasks waiting for the GPU (default 16)
- `--max-batch-target-tokens`: cap on estimated generated audio tokens
- `--max-batch-conditioning-tokens`: cap on estimated prompt/context tokens
- `--max-batch-padding-ratio`: guardrail against inefficient padding-heavy batches
- `--clone-prompt-cache-size`: number of prepared clone prompts to cache
- `--registered-clone-prompt-store-size`: number of registered prompt IDs to keep
- `--token-estimate-cache-size`: LRU size for tokenizer-based batch estimates
- `--decode-postprocess-workers`: CPU worker threads for audio postprocessing
- `--max-cuda-graphs`: maximum CUDA Graph shapes retained by `faster` and
  `hybrid`; default `8`, and `0` disables graph caching

When a pending queue is full, the request returns HTTP 429. This is overload
protection, not a latency or successful-concurrency guarantee. `/health` exposes
`batcher_queue` with both pending counts, their limits, and the oldest prepared
request's wait time. These counts exclude executing work and requests waiting for
a synchronous HTTP handler thread. Reference encoding uses `torch.inference_mode()`
inside the GPU owner thread; PyTorch gradient modes are local to each thread.

The launcher accepts `MAX_PENDING_REQUESTS` and `MAX_PENDING_GPU_TASKS` as well as
the existing batching environment settings. `/generate` still returns a complete
WAV. Sentence splitting generates each sentence and joins every result before
returning; it does not send partial audio to the client.

Typical startup:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --ip 0.0.0.0 --port 8002
```

Faster and safer startup if you always supply `ref_text` for clone mode:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --no-asr --ip 0.0.0.0 --port 8002
```

With this startup mode, clone requests that omit `ref_text` return `400`
instead of loading Whisper ASR onto the GPU during request handling. Use
`--allow-lazy-asr` only when that VRAM tradeoff is intentional.

Save generated files locally too:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --port 8002 --save-dir ./generated_wavs
```

## Endpoints

### `GET /healthz`

Returns a compact readiness response once the model and batcher are ready
(`503` while loading, or with `"error": "cuda_fault"` after a CUDA failure):

```json
{
  "ok": true,
  "service": "omnivoice",
  "worker_type": "tts",
  "status": "healthy"
}
```

### `GET /health`

Returns runtime status (with HTTP `503`, `"status": "error"`, and a
`cuda_fault` reason after a CUDA failure):

- model checkpoint
- runner
- device
- dtype
- sample rate
- whether startup ASR loading was requested
- whether ASR is currently loaded
- whether a save directory is configured
- whether SageAttention is enabled

Example:

```bash
curl http://127.0.0.1:8002/health
```

Typical response:

```json
{
  "status": "ok",
  "model": "k2-fsa/OmniVoice",
  "runner": "hybrid",
  "device": "cuda",
  "dtype": "fp16",
  "sample_rate": 24000,
  "load_asr": false,
  "asr_loaded": false,
  "save_dir": null,
  "sage_attention": false
}
```

### `GET /languages`

Returns the supported language list derived from upstream OmniVoice language
metadata.

Each item includes:

- `id`
- `name`
- `display_name`

Example:

```bash
curl http://127.0.0.1:8002/languages
```

### `POST /generate`

Generates speech and returns a WAV file.

The endpoint expects multipart form data.

Response:

- HTTP `200`
- `Content-Type: audio/wav`
- inline WAV bytes in the response body

Batch-aware response headers now also include:

- `X-OmniVoice-Queue-Wait-Ms`
- `X-OmniVoice-Batch-Exec-Ms`
- `X-OmniVoice-Batch-Requests`
- `X-OmniVoice-Batch-Target-Tokens`
- `X-OmniVoice-Batch-Conditioning-Tokens`
- `X-OmniVoice-Batch-Max-Sequence-Length`
- `X-OmniVoice-Batch-Lane`

## Gradio Parity

This API was checked against the upstream Gradio UI in:

- [.venv/lib/python3.12/site-packages/omnivoice/cli/demo.py](/workspace/omnivoice-triton/.venv/lib/python3.12/site-packages/omnivoice/cli/demo.py)

The API matches the Gradio generation controls for:

- `text`
- `language`
- `instruct`
- `ref_audio`
- `ref_text`
- `num_step`
- `guidance_scale`
- `denoise`
- `speed`
- `duration`
- `preprocess_prompt`
- `postprocess_output`

Behavior parity notes:

- `design` mode allows an empty `instruct`, matching Gradio's fallback behavior
  when no voice-design attributes are selected
- clone prompt preparation can be reused across requests when the uploaded
  reference audio bytes, `ref_text`, and `preprocess_prompt` match

## Batching Behavior

The server batches at the HTTP layer and relies on upstream OmniVoice's native
list-based `generate(...)` path.

Current lane split:

- `short_noref`
- `short_ref`
- `long_noref`
- `long_ref`

Requests only batch together when these fields match:

- `num_step`
- `guidance_scale`
- `t_shift`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`
- `denoise`
- `postprocess_output`
- `audio_chunk_duration`
- `audio_chunk_threshold`
- batch lane

Requests can still vary within a batch on:

- `text`
- `language`
- `instruct`
- clone prompt identity
- `speed`
- `duration`

Current limitation:

- batching is still single-process and anchor-based, so one long batch can still
  delay later requests until that merged call finishes
- `clone` mode allows optional `instruct`, matching the Gradio clone tab
- `duration <= 0` is treated as unset, matching Gradio's behavior
- `language="Auto"` or blank becomes `None`

Intentional API-only extras:

- `auto` mode is exposed directly
- advanced decoding parameters are accepted even though Gradio does not show
  them in the UI:
  - `t_shift`
  - `layer_penalty_factor`
  - `position_temperature`
  - `class_temperature`

One structural difference:

- Gradio has separate dropdowns for voice-design categories
- the HTTP API accepts the final assembled `instruct` string instead

This is equivalent to what the Gradio app eventually passes into
`model.generate(...)`.

## Request Modes

### `mode=auto`

Text-only generation.

Allowed:

- `text`
- optional `language`
- optional generation settings

Rejected:

- `instruct`
- `ref_audio`
- `ref_text`

Example:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Hello from omnivoice-triton' \
  -o auto.wav
```

### `mode=design`

Voice design from instruction text.

Allowed:

- `text`
- optional `instruct`
- optional `language`
- optional generation settings

Rejected:

- `ref_audio`
- `ref_text`

Important behavior:

- if `instruct` is omitted or blank, generation falls back to text-only style
  conditioning, matching the Gradio design tab when no attributes are selected

Example with explicit design attributes:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from omnivoice-triton' \
  -F instruct='female, young adult, high pitch' \
  -o design.wav
```

Example with no selected attributes:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from omnivoice-triton' \
  -o design_auto_like.wav
```

### `mode=clone`

Voice cloning from a reference audio clip.

Required:

- `text`
- `ref_audio`

Optional:

- `ref_text`
- `instruct`
- `language`
- generation settings

Behavior:

- the uploaded file is written to a temporary file
- the server calls `model.create_voice_clone_prompt(...)`
- the resulting prompt is passed into `model.generate(...)`
- if `ref_text` is omitted and ASR is already loaded, OmniVoice may use ASR to
  auto-transcribe
- if `--no-asr` was used, clone requests without `ref_text` are rejected unless
  the server was also started with `--allow-lazy-asr`

Example:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from omnivoice-triton' \
  -F ref_audio=@/full/path/to/ref.wav \
  -F ref_text='Hello from omnivoice-triton' \
  -o clone.wav
```

Clone with extra style control:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from omnivoice-triton' \
  -F ref_audio=@/full/path/to/ref.wav \
  -F ref_text='Hello from omnivoice-triton' \
  -F instruct='whisper' \
  -o clone_whisper.wav
```

## Request Fields

`/generate` accepts these form fields:

- `mode`
- `text`
- `language`
- `instruct`
- `ref_text`
- `num_step`
- `guidance_scale`
- `t_shift`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`
- `speed`
- `duration`
- `denoise`
- `preprocess_prompt`
- `postprocess_output`
- `ref_audio`

### Field Semantics

`text`

- required
- blank strings are rejected

`language`

- optional
- blank, `Auto`, or `None` are normalized to `None`
- otherwise passed through to OmniVoice language resolution

`instruct`

- optional in `design`
- optional in `clone`
- rejected in `auto`

`num_step`

- default: `32`
- must be an integer greater than `0`

`guidance_scale`

- default: `2.0`
- must be `>= 0`

`t_shift`

- default: `0.1`
- must be `>= 0`

`layer_penalty_factor`

- default: `5.0`
- must be `>= 0`

`position_temperature`

- default: `5.0`
- must be `>= 0`

`class_temperature`

- default: `0.0`
- must be `>= 0`

`speed`

- optional
- must be `> 0` if provided
- forwarded only when it differs from `1.0`

`duration`

- optional
- if omitted, normal duration estimation is used
- if `<= 0`, it is treated as unset to match Gradio behavior
- if positive, it overrides duration estimation

`denoise`

- default: `true`

`preprocess_prompt`

- default: `true`
- mainly affects clone mode prompt cleanup

`postprocess_output`

- default: `true`

`ref_audio`

- required in clone mode
- ignored or rejected in the other modes

## Example Requests

### Auto Mode

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Hello from the omnivoice triton API server.' \
  -o auto.wav
```

### Design Mode

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from the omnivoice triton API server.' \
  -F instruct='female, young adult, high pitch' \
  -o design.wav
```

### Clone Mode

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from the omnivoice triton API server.' \
  -F ref_audio=@/full/path/to/reference.wav \
  -F ref_text='Hello from the omnivoice triton API server.' \
  -o clone.wav
```

### Tuned Generation Example

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='This is a slower and more deliberate sample.' \
  -F instruct='male, low pitch' \
  -F num_step=24 \
  -F guidance_scale=2.0 \
  -F speed=0.9 \
  -F denoise=true \
  -F preprocess_prompt=true \
  -F postprocess_output=true \
  -o design_tuned.wav
```

### Inspect Response Headers

```bash
curl -i -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Header check example' \
  -o header_test.wav
```

## Listening To Output

On Linux:

```bash
ffplay -autoexit auto.wav
ffplay -autoexit design.wav
ffplay -autoexit clone.wav
```

Fallback:

```bash
aplay auto.wav
aplay design.wav
aplay clone.wav
```

## Response Headers

Successful `/generate` responses include:

- `Content-Disposition: inline; filename="omnivoice.wav"`
- `X-OmniVoice-Request-Id`
- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF` if audio duration is non-zero
- `X-OmniVoice-Runner`
- `X-OmniVoice-Device`
- `X-OmniVoice-Peak-Vram-Gb`
- `X-OmniVoice-Saved-Path` when `--save-dir` is enabled

### What `RTF` Means

`RTF` is real-time factor:

```text
RTF = generation_time_seconds / output_audio_seconds
```

Interpretation:

- `RTF < 1.0`: faster than real-time
- `RTF = 1.0`: real-time
- `RTF > 1.0`: slower than real-time

## Save Directory

If `--save-dir` is set:

- the WAV response is still returned inline
- a copy is also persisted locally
- the saved path is added to the response headers

Filename format:

```text
{timestamp}_{mode}_{slug}_{request_id}.wav
```

## Concurrency Model

This server currently uses:

- one in-process runner
- one loaded OmniVoice model
- one `threading.Lock()` around generation

That means requests are serialized within a single server process.

This is good for:

- correctness
- smoke tests
- latency testing
- simple single-GPU app hosting

It is not designed for:

- high throughput
- multi-request batching
- streaming audio

## Error Handling

The server maps common failures to HTTP responses:

- validation issues: `400`
- upstream `ValueError`: `400`
- upstream `RuntimeError`: `500`
- model not yet ready: `503`
- unexpected exceptions: `500`

## Clone Troubleshooting

The most common clone issue is not a server bug. It is a local file path issue
in `curl`.

If you see:

```text
curl: (26) Failed to open/read local data from file/application
```

that means `curl` could not open the file after `@...`.

Examples:

```bash
-F ref_audio=@reference.wav
```

only works if `reference.wav` exists in the current directory.

Safer version:

```bash
-F ref_audio=@/full/path/to/reference.wav
```

Check the file first:

```bash
ls -l /full/path/to/reference.wav
```

Common gotchas:

- the file is not in your current directory
- the filename is different from what you typed
- the path contains spaces and was not quoted correctly
- you used `~` and shell expansion did not behave the way you expected

## Recommended Testing Workflow

For first-pass validation:

1. start the server
2. hit `/health`
3. send one `auto` request
4. send one `design` request
5. send one `clone` request with a known-good local WAV
6. inspect:
   - HTTP status
   - WAV playback
   - latency headers
   - server logs

Example:

```bash
curl http://127.0.0.1:8002/health
```

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Testing latency logging' \
  -o smoke.wav
```

Look for:

- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF`

## Current Limitations

- no request batching
- no audio streaming
- one serialized generation lock
- one runner instance per process
- clone uploads are written to temporary files before prompt construction

These are acceptable for practical testing and lightweight deployment, but not
for high-concurrency production scheduling.
