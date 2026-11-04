# Complete-audio concurrency pilot

Hardware: **unverified until execution**. This directory contains an executable
protocol, not a GPU performance result. Each run records the exact NVIDIA GPU,
physical VRAM from `nvidia-smi`, visible VRAM from PyTorch, driver/CUDA runtime,
package versions, co-resident GPU processes, source file hashes, checkpoint
revision and date in `metadata.json`.

`run.py` starts one fresh, task-owned OmniVoice HTTP process on `127.0.0.1`, runs
the finite workload, then terminates only that process group. Run it with the
GPU's TTS virtualenv Python so the server and hardware collector use the same
dependencies. It does not rent instances, register a production worker, or invoke
paid transcription services. Lingua callback environment variables are removed
from the child server environment.

```sh
/workspace/tts/.venv/bin/python bench/run.py \
  --source /workspace/source-baseline \
  --source-ref EXACT_COMMIT \
  --model /workspace/pinned-model \
  --model-revision EXACT_MODEL_COMMIT \
  --label before \
  --output /workspace/results/before
```

Repeat with the candidate source and a new output directory. Existing output
directories are rejected to prevent replacing earlier audio. `--smoke` sends just
one generation after registering the three reference fixtures.

Default model profile: hybrid runner, FP16, no SageAttention, 16 generation
steps, speed 1.0, denoise enabled, postprocessing off, no sentence splitting.
Batch settings: 5 ms collection, maximum 8 requests, 4096 target and conditioning
token budgets, 16 pending generations, 4 pending GPU tasks, 8 retained CUDA graphs.
The CLI exposes batch parameters for clearly labeled follow-up experiments.

Every response is one **complete WAV**. The client measures until the full HTTP
response body arrives. This loopback measurement excludes the external network,
Lingua gateway, MuseTalk rendering, and client playback.

The full workload has 170 generation attempts per arm:

| Phase | Requests | Submission |
| --- | ---: | --- |
| Warmup, short EN/ES/AR | 3 | Sequential |
| Warmup, English short | 10 | Simultaneous |
| EN/ES/AR × short/medium/long | 9 | Sequential |
| English short repeats | 10 | Sequential |
| English short | 10 | Simultaneous |
| Mixed corpus plus one English short | 10 | Simultaneous |
| English short overload probe | 100 | Simultaneous |
| Mixed corpus twice | 18 | One arrival every 2 seconds |

Warmups are labeled and retained separately. Variable shapes can evict CUDA
graphs; later phases are not asserted to be universally warm. The API exposes no
per-request seed: source inputs/order are matched, but this protocol does not
claim paired seeds or bit-identical waveforms. The language fixtures are original
12–15 second recordings retained from the prior study. They exceed upstream's
3–10 second guidance and are challenging fixed fixtures, not a recommendation.

HTTP 429 is an overload rejection and does not establish successful concurrency
capacity. Successful-response p95 is explicitly conditional on HTTP 200; all
rejections, failures and timeouts remain in the denominator of outcome counts.
The harness stops after unexpected HTTP/runtime failures. It makes no claim
about 1000/5000-call capacity. Per-request and per-phase deadlines bound a run.

`requests.jsonl` records every completed attempt, full-response latency, server
timing headers, parameters and signal measurements. Every HTTP 200 body is saved
in a unique `audio/PHASE-INDEX-CASE.wav` with SHA-256. Signal validity establishes
neither word accuracy nor perceptual quality. `quality-manifest.json` uses paths
relative to its parent directory and carries language/expected text only for
**post-recognition scoring** by the separate offline ASR stage. Keep the arm
directory structure intact when retaining/downloading evidence.

`health-probes.jsonl` samples `/health` during load, including response times,
timeouts and queue snapshots. `summary.json` records each phase. `outcome.json`
records whether the finite workload completed. `server.log` and final health are
retained even on failure. Run ASR after TTS timing, in a separate GPU phase, to
avoid contaminating latency measurements.

Static validation: Python compile succeeded. A CPU-only fake HTTP smoke checked
all three reference registrations, successful full-WAV retention, relative
manifest paths and separate rejection accounting. That smoke used no model and
is not GPU inference evidence.
