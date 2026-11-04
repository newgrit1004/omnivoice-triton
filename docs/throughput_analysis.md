# Throughput Analysis

## Goal

Target: support **100 concurrent requests** with **3-5 seconds wall time max** while keeping:

- `num_step=16`
- high-quality audio as a hard constraint
- `hybrid` as the primary serving path unless a faster path clearly preserves quality

This document is a repo-specific optimization plan based on:

- the current FastAPI serving path in `src/omnivoice_triton/cli/api_server.py`
- the batcher in `src/omnivoice_triton/serving/batching.py`
- the runner stack in `src/omnivoice_triton/models/`
- upstream OmniVoice generation behavior
- stored load-test artifacts
- live load tests on an RTX 3090 on 2026-04-17
- live load tests on an RTX 5070 on 2026-04-19

## Reality Check

Current behavior does **not** meet the target yet.

Observed live results on RTX 3090 with warmed `hybrid` runner:

- `clone`, `num_step=16`, 100 concurrent requests: about **30.9s** wall
- `design`, `num_step=16`, 100 concurrent requests: about **12.1s** wall
- `design`, `num_step=8`, 100 concurrent requests: about **5.7s** wall

Observed live results on RTX 5070 with warmed `hybrid` runner:

- `design`, `num_step=16`, 1 request at concurrency 1: about **0.35s** wall
  - about **2.86 req/s** for the one-request run
  - avg latency about **124ms**
  - avg queue wait about **10ms**
  - avg merged batch execution about **112ms**
  - per-response batch sizes: `1=>1`
  - peak VRAM about **6.40 GB**
- `design`, `num_step=16`, 25 concurrent requests: about **2.69s** wall
  - about **9.29 req/s**
  - avg latency about **1706ms**, p50 about **1687ms**, p95 about **2276ms**
  - avg queue wait about **667ms**
  - avg merged batch execution about **1010ms**
  - per-response batch sizes: `1=>1`, `2=>4`, `4=>4`, `16=>16`
  - peak VRAM about **6.40 GB**
- `design`, `num_step=16`, 50 concurrent requests: about **5.06s** wall
  - about **9.89 req/s**
  - avg latency about **2937ms**, p50 about **3206ms**, p95 about **3950ms**
  - avg queue wait about **765ms**
  - avg merged batch execution about **2128ms**
  - per-response batch sizes: `2=>2`, `8=>16`, `32=>32`
  - peak VRAM about **6.40 GB**
- `design`, `num_step=16`, 100 concurrent requests: about **11.52s** wall
  - about **8.68 req/s**
  - avg latency about **3491ms**, p50 about **3747ms**, p95 about **4181ms**
  - avg queue wait about **1063ms**
  - avg merged batch execution about **2383ms**
  - per-response batch sizes: `4=>4`, `8=>16`, `16=>16`, `32=>64`
  - avg peak VRAM about **6.10 GB**, max observed peak VRAM about **6.97 GB**

Stored repo results are better than the live 3090 run, but still not within target for clone mode:

- `results_100_clone_step16.csv`: about **16.0s** wall for 100 clone requests

Implication:

- On the RTX 5070, `design` mode scales reasonably up to around **25 concurrent requests** and nearly holds the target at **50 concurrent requests**:
  - `25` concurrent lands inside the `3-5s` target window at **2.69s**
  - `50` concurrent narrowly misses the `5s` upper bound at **5.06s**
  - `100` concurrent still misses badly at **11.52s**
- Throughput on the RTX 5070 improves slightly from `25` to `50` concurrent, then drops at `100`:
  - about **9.29 req/s** at `25`
  - about **9.89 req/s** at `50`
  - about **8.68 req/s** at `100`
- The drop from `50` to `100` concurrent, combined with the rising queue wait, suggests the single in-process merged batch path is saturating:
  - avg queue wait grows from about **667ms** at `25`
  - to about **765ms** at `50`
  - to about **1063ms** at `100`
- The RTX 5070 `design`, `num_step=16`, `100`-concurrent run is only a **modest improvement** over the live RTX 3090 baseline:
  - wall time improved from **12.1s** to **11.52s**
  - that is about **0.58s faster** or about **4.8% lower wall time**
  - throughput improved from about **8.26 req/s** to **8.68 req/s**, about **5.0%**
- Single-request performance on the warmed RTX 5070 is strong for local latency testing:
  - about **0.35s** wall time for one request
  - about **124ms** server-side latency
  - about **10ms** queue wait
- **`design` / no-reference mode** can meet the target at lower concurrency on the RTX 5070, but current single-replica results are still not close enough at the actual goal of `100` concurrent requests with `num_step=16`.
- **`clone` mode at `num_step=16`** likely needs more than small tuning. It will require major serving improvements and may still need stronger hardware or multi-replica scheduling to consistently hit 3-5 seconds.

## Main Bottlenecks

### 1. Single in-flight merged batch

The server still executes only **one `model.generate(...)` call at a time**.

Impact:

- 100 concurrent requests drain in sequential waves.
- One large batch blocks every later request.
- Tail latency is dominated by queueing, not just model speed.

### 2. Batch-size instability hurts CUDA Graph reuse

CUDA Graph capture is keyed by exact input shape.

Impact:

- burst traffic produces multiple graph captures for different batch sizes
- cold-path graph capture appears during live traffic
- shape churn adds avoidable overhead

### 3. Clone prompt preparation is still expensive

Clone requests still do meaningful CPU work before they reach the batcher:

- audio decode from upload bytes
- silence trimming
- resampling
- optional ASR
- audio tokenizer encode

The current clone-prompt cache stores tokens on CPU and the generation path copies them back to GPU per request.

### 4. Duplicate work before generation

The API estimates batching cost by tokenizing style/text in the server, then upstream OmniVoice tokenizes again during input preparation.

Impact:

- extra CPU overhead for every request
- extra latency before requests even enter merged execution

### 5. Decode and post-process run after token generation

After token generation, each item still pays:

- audio-token decode
- silence removal
- fade/pad work
- WAV serialization

This is especially costly in large response bursts.

### 6. Default `hybrid` path is not the maximum-speed variant

`triton` and `hybrid` patch only layers `0..23` by default and leave the last 4 layers unpatched for quality safety.

That is a sensible default, but it means there is still some model-side speed headroom to measure.

## Strategy

The fastest path to the goal is not "make batches bigger".

On the RTX 3090, increasing `max_batch_requests` from 32 to 64 made throughput worse because larger merged batches increased `batch_exec_ms` more than they improved utilization.

So the right plan is:

1. stabilize batch shapes
2. remove per-request CPU overhead before batching
3. reduce post-generation overhead
4. benchmark more aggressive model-side optimizations while keeping `num_step=16`
5. only then decide whether a single model replica can meet the target

## Implementation Status

Implemented in repo on **2026-04-17**:

- **Phase 0**
  - per-request stage timing headers for:
    - pre-batch work
    - prompt preparation
    - batch estimate work
    - queue wait
    - batch execution
    - response encoding
  - richer `scripts/load_test_api.py` summary output
  - CUDA Graph capture metrics exposed from health checks
  - GPU utilization and VRAM telemetry exposed in health checks, server logs,
    and load-test CSV/summary output
- **Phase 1**
  - request-count batch buckets in the batcher
  - configurable `--batch-bucket-sizes`
  - startup CUDA Graph prewarm controls:
    - `--prewarm-clone-batch-sizes`
    - `--prewarm-clone-sequence-lengths`
- **Phase 2**
  - `POST /clone-prompts`
  - `prompt_id` support on `POST /generate`
  - reusable registered clone prompt store for throughput-sensitive clone traffic
- **Phase 3**
  - tokenizer estimate LRU cache for server-side batching estimates
- **Phase 5**
  - `--full-triton-patch` to benchmark full-layer Triton patching at `num_step=16`
  - model-internal timing breakdown surfaced through health, response headers,
    batch logs, and load-test CSV/summary output
  - current measured clone results on the RTX 3090 for `100` requests /
    `100` concurrency / `num_step=16`:
    - baseline hybrid: about `19.69s`
    - hybrid + clone graph prewarm: about `17.46s`
    - hybrid + full-layer Triton + clone graph prewarm: about `17.18s`
- **Phase 4**
  - throughput-oriented postprocess modes: `full`, `light`, and `off`
  - threaded CPU-side decode/postprocess execution with detailed timing split
  - current measured clone results on the RTX 3090 for `100` requests /
    `100` concurrency / `num_step=16`:
    - full postprocess path: about `17.13s`
    - light postprocess path: about `16.16s`
    - full path decode split: about `463ms` audio decode + `131ms`
      post-audio cleanup
    - light path decode split: about `462ms` audio decode + `7ms`
      post-audio cleanup

Still planned and not implemented yet:

- **Phase 6** smarter scheduler / multi-worker drain logic
- **Phase 7** multi-replica scaling

## Optimization Phases

## Phase 0: Lock Baseline And Instrumentation

Status: **Implemented**

Goal: make throughput work measurable and repeatable.

Changes:

- keep `scripts/load_test_api.py` as the standard load harness
- define standard scenarios:
  - `design`, `num_step=16`, 100 requests, concurrency 100
  - `clone`, `num_step=16`, 100 requests, concurrency 100
- capture:
  - wall time
  - p50/p95 latency
  - queue wait
  - batch execution time
  - batch-size distribution
  - graph-capture count by shape
- log separate timings for:
  - request preprocessing
  - prompt-cache hit or miss
  - merged `model.generate(...)`
  - decode/postprocess
  - WAV serialization

Why first:

- we need clean before/after data for each phase
- current headers mostly show batch timing, but not enough sub-stage breakdown

Success criteria:

- every load test run produces a clear stage-level timing breakdown
- graph-capture churn is visible in logs or metrics

## Phase 1: Stabilize Batch Shapes For CUDA Graph Reuse

Status: **Implemented**

Goal: stop paying graph-capture and shape-churn penalties during burst traffic.

Changes:

- introduce **batch buckets** for short-lane traffic, for example:
  - `8`
  - `16`
  - `32`
- prefer filling the nearest bucket instead of emitting arbitrary sizes like `18`, `22`, `39`, `40`
- optionally hold requests slightly longer when a batch is near the next bucket boundary
- prewarm graph shapes at startup for the common buckets and common
  sequence-length bands
- consider separate lanes for:
  - `short_noref`
  - `short_ref`
  - `medium_ref`
  instead of only short/long

Expected impact:

- fewer first-use graph captures during live bursts
- steadier `batch_exec_ms`
- better p95 and p99 latency

Priority:

- **Very high**

Success criteria:

- common burst runs only use a small, fixed set of graph shapes
- graph capture happens at startup or warmup, not mid-run

## Phase 2: Add A Fast Clone Prompt Path

Status: **Implemented**

Goal: remove repeated clone prompt prep from the request hot path.

Changes:

- add a **prompt registration** workflow:
  - `POST /clone-prompts`
  - return a `prompt_id`
  - `POST /generate` can use `prompt_id` instead of raw `ref_audio`
- keep registered prompts in:
  - GPU memory if capacity allows, or
  - pinned CPU memory if not
- avoid re-decoding and re-tokenizing the same reference audio for repeated callers
- keep the current upload-based flow as fallback, but make `prompt_id` the throughput path
- require `ref_text` for throughput-sensitive clone serving so ASR never lands on the hot path

Expected impact:

- major reduction in clone-mode CPU overhead
- less variance before requests enter the batcher
- less GPU upload churn for cached prompts

Priority:

- **Very high**

Success criteria:

- repeated clone traffic can avoid raw upload decode and prompt creation entirely
- prompt-cache hits show materially lower pre-batch latency

## Phase 3: Remove Duplicate Server-Side Tokenization

Status: **Implemented**

Goal: stop doing token-count estimation work twice.

Changes:

- avoid full tokenizer calls in the server just to estimate conditioning size
- either:
  - cache estimated token counts alongside prompt data, or
  - move the estimate into a lighter-weight heuristic, or
  - have upstream-prepared metadata returned with the prompt object
- keep batching heuristics approximate if needed; they do not need exact token counts to be useful

Expected impact:

- smaller but consistent CPU savings
- lower request admission latency

Priority:

- **High**

Success criteria:

- design and clone requests do not fully tokenize text twice before generation

## Phase 4: Reduce Decode And Postprocess Cost

Status: **Implemented**

Goal: keep the GPU focused on generation and reduce serial tail work.

Changes:

- add a throughput-oriented output mode:
  - `postprocess_output=light`
  - lighter silence handling or skip silence removal
- move WAV serialization off the critical path where possible
- benchmark whether decode/postprocess should run in a small worker pool after `model.generate(...)`
- if quality remains acceptable, reduce or disable expensive silence stripping for high-throughput server mode

Notes:

- this phase must preserve audible quality
- the goal is not to change the voice, only to trim avoidable post-work

Expected impact:

- better tail latency
- better burst completion time for large batches

Priority:

- **High**

Success criteria:

- decode/postprocess is no longer a large share of total per-response time

Current measured update on 2026-04-17:

- with `hybrid + full-layer Triton + clone graph prewarm`, the default `full`
  postprocess path completed the `100` request clone burst in about `17.13s`
- the same workload with `postprocess_mode=light` completed in about `16.16s`
- `decode_postprocess_ms` dropped from about `594ms` in `full` mode to about
  `469ms` in `light` mode
- most of the remaining decode/postprocess cost is still audio decode itself:
  - `full`: about `463ms` audio decode + `131ms` post-audio cleanup
  - `light`: about `462ms` audio decode + `7ms` post-audio cleanup
- this means lighter cleanup helps, but the bigger remaining hot paths are
  still iterative generation and the decode step itself

Important:

- `light` mode should get listening-based QA before it becomes the default
  clone throughput preset
- Phase 4 reduced the CPU-side tail, but it did not change the compute-bound
  nature of the main generation path

## Phase 5: Benchmark The Fastest Safe Model Variant At `num_step=16`

Status: **Partially Implemented**

Goal: squeeze more speed out of the model path without dropping below your quality bar.

Changes to benchmark:

- `patch_range=None` for full-layer Triton patching
- `hybrid + SageAttention` on Ampere
- `bf16` versus `fp16` if numerically stable on the target GPU
- prewarmed graph shapes for the dominant serving buckets

Current measured update on 2026-04-17:

- prewarming the dominant clone graph shapes reduced average `batch_exec_ms`
  from about `4009ms` to about `3766ms`
- adding full-layer Triton patching on top of prewarm reduced average
  `batch_exec_ms` again to about `3698ms`
- decode/postprocess stayed roughly flat at about `555-563ms`, so the current
  wins are mostly from the iterative generation path rather than postprocessing

Important:

- this phase is benchmark-driven, not assumption-driven
- full-layer patching may help throughput, but it needs quality verification

Priority:

- **High**

Success criteria:

- pick one serving preset that is fastest while preserving acceptable audio quality at `num_step=16`

## Phase 6: Smarter Scheduler Instead Of Bigger Batches

Status: **Planned**

Goal: improve drain time without creating giant slow batches.

Changes:

- keep batch caps around the best local knee instead of pushing to 64+
- add scheduler logic that prefers:
  - stable buckets
  - same lane
  - same prompt class
  - low padding waste
- optionally let one nearly-full batch launch while a second batch continues collecting if enough queued work exists

Notes:

- this may require more than one worker thread or more than one model replica
- the current one-worker architecture is the structural wall

Priority:

- **Medium to high**

Success criteria:

- wall time improves without increasing p95 batch execution time

## Phase 7: Scale Beyond One In-Flight Generate

Status: **Planned**

Goal: decide whether the 3-5 second target is possible on one GPU, and if not, add the minimum necessary parallelism.

Options:

- two model replicas on one large GPU if memory allows
- process-level sharding with separate queues
- multi-GPU serving when clone throughput is the real product target

Reality:

- if clone mode at `num_step=16` still sits well above target after Phases 1-6, **single in-process single-replica serving is the limit**, not just an implementation bug

Priority:

- **Conditional but likely necessary for clone mode**

Success criteria:

- sustained 100-request burst completion within 3-5 seconds in the actual target mode

## Best Order To Execute

Recommended order:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

That order gave the best chance of gaining real throughput early without
rewriting too much too soon. After the current work, the highest-value
remaining items are Phase 6 first, then Phase 7 if the wall target is still
hard.

## Recommended Product Modes

To keep `num_step=16` while maximizing throughput, define explicit serving presets.

### Preset A: High-Quality Design

- `mode=design`
- `num_step=16`
- prewarmed common graph buckets
- full request batching
- lightweight postprocess if quality is preserved

Target:

- strongest candidate for the 3-5 second goal on one strong GPU

### Preset B: High-Quality Clone

- `mode=clone`
- `num_step=16`
- `prompt_id` optional but recommended when the caller can reuse a prompt
- `ref_text` required
- no ASR in hot path
- cached prompt tokens on GPU or pinned memory
- `postprocess_mode=full` for quality-first serving
- `postprocess_mode=light` only after listening-based validation

Target:

- likely needs the full Phase 1-7 roadmap

## Acceptance Gates

Do not call the target met unless all are true:

- 100 concurrent requests finish in **<= 5s wall**
- p95 latency stays within the same bound class as wall target
- no on-demand ASR in the hot path
- no graph captures during warmed steady-state runs
- audio quality remains acceptable at `num_step=16`

## Testing

Server regression tests:

- `.venv/bin/pytest tests/test_api_server.py -q`

Recommended throughput validation commands after warm server startup:

- design baseline:
  - `.venv/bin/python scripts/load_test_api.py --mode design --num-step 16 --requests 100 --concurrency 100 --warmup-requests 8 --csv results_design_100.csv`
- clone with raw upload path:
  - `.venv/bin/python scripts/load_test_api.py --mode clone --num-step 16 --requests 100 --concurrency 100 --warmup-requests 4 --ref-audio clone.wav --ref-text "Reference text" --csv results_clone_upload_100.csv`
- clone with registered prompt fast path:
  - `.venv/bin/python scripts/load_test_api.py --mode clone --num-step 16 --requests 100 --concurrency 100 --warmup-requests 4 --register-clone-prompt --ref-audio clone.wav --ref-text "Reference text" --csv results_clone_prompt_id_100.csv`
- clone with the lighter postprocess path:
  - `.venv/bin/python scripts/load_test_api.py --mode clone --num-step 16 --requests 100 --concurrency 100 --warmup-requests 4 --ref-audio clone.wav --postprocess-mode light --csv results_clone_light_100.csv`

What to compare between runs:

- wall time
- latency p50/p95
- `pre_batch_ms`
- `prompt_prepare_ms`
- `batch_estimate_ms`
- `queue_wait_ms`
- `batch_exec_ms`
- `response_encode_ms`
- batch-size distribution
- prompt-source distribution

## Bottom Line

The best optimization opportunities are not the Triton kernels themselves. The repo already got strong wins there.

The next big gains come from:

- **stable batch shapes**
- **pre-registered clone prompts**
- **less request-side CPU work**
- **less audio decode cost and lighter post-audio cleanup when quality holds**
- **benchmarking the fastest safe `hybrid` variant at `num_step=16`**

For `design` mode, the 3-5 second goal looks aggressive but plausible with strong serving work.

For `clone` mode, the same goal is a stretch on the current single-replica architecture and may require either:

- a stronger GPU than the current 3090
- multiple replicas
- or both
