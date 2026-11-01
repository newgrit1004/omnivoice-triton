# Complete-audio concurrency audit and scaling plan — 2026-11-01

## Scope and evidence

The current GPU pilot uses the existing **NVIDIA GeForce RTX 3060**, Vast instance
`58349253`: NVIDIA-SMI reports **12,288 MiB**, 170 W and driver **535.161.07**;
PyTorch reports **12,629,508,096 visible bytes**. Host CPU: AMD Ryzen 7 5800X,
16 logical CPUs. Runtime: Python 3.12.3, Torch 2.8.0+cu128, Triton 3.4.0,
OmniVoice 0.1.4, Transformers 5.5.4, Starlette 1.0.0. Evidence is live
NVIDIA-SMI, PyTorch device properties, package metadata and HTTP responses on
2026-11-01. This production worker was not isolated from host contention.

The repository audit and lint checks ran on an Apple M5 Pro without CUDA inference.
The 61 API/batcher regression tests used a fake model on the 3060 host, with CUDA
visible for the existing CUDA-storage checks; these were not real-model generation
tests. Earlier CPU-hidden attempts failed the API's CUDA availability check and
were rerun with CUDA visible. New real-model measurements are separate below.

The user requires one **complete WAV** before handoff to MuseTalk. Internal
splitting is permitted only if every piece is assembled before returning. There
is no audio streaming implementation in this plan. Target: p95 complete-audio
response below 500 ms, for realistic calls and simultaneous bursts at
10, 100, 1,000 and 5,000. Network, gateway and MuseTalk latency must be measured
separately; a loopback TTS result is not end-to-end call latency.

## Repository and deployment identity

- User fork: `https://github.com/AhmadAFS1/omnivoice-triton.git`.
- Pulled `main` with `git pull --ff-only`: `4c93cbd` →
  `fea6af0e068ef4e9c7a77a195cd33fad06b10df3`.
- The live 3060 initially ran that same latest `main`, runner **hybrid**.
- The user's unmerged `fix/cuda-graph-capture-race` branch adds
  `e4495e08a44b80d3571ead172868c8ea1860ec4f` on top of latest main.
- New branch `codex/complete-audio-concurrency` includes that fix and
  `4d37b0ac48e339cada6242da8d44547bf5d77c48` for bounded queues and reference
  inference mode. This code commit is deployed on the existing 3060. It has not
  been pushed or merged into remote main.
- `hybrid` means this fork's Triton kernels plus CUDA Graph replay. It still
  loads upstream OmniVoice weights/code. It is not the unmodified base runner.
  SageAttention is off. The installed upstream dependency remains 0.1.4;
  pulling the fork does not upgrade that dependency.

## What concurrency work already exists

| Area | Implementation found | Limit for low-latency calls |
| --- | --- | --- |
| HTTP microbatching | `GenerationBatcher`, one background thread, compatible requests merged into `model.generate(list)` | One merged batch runs at a time; this is not continuous token scheduling |
| Compatibility lanes | Short/long × reference/no reference, plus matching generation settings | Different lanes/settings fragment batches; longest work delays its batch |
| Token and padding budgets | Target-token sum, conditioning-token sum, max batch count, padding ratio | Estimates only; the first selected request bypasses these aggregate token budgets |
| Batch size buckets | Default 1/2/4/8/16/32; trim to an available smaller bucket | Reduces batch shapes, but can defer a remainder; sequence lengths remain variable |
| CUDA Graph cache | Exact input-shape LRU, eight graphs by API default | No VRAM byte budget; different text/reference lengths cause capture churn |
| Triton kernels | Fused normalization/SwiGLU and graph replay in hybrid | Optimizes execution, not ingress queueing or fleet routing |
| Clone reference cache | SHA-256 + transcript + preprocessing key, CPU prompt LRU, per-key in-flight deduplication | Duplicate callers block on an event; unique cold voices still require GPU encoding |
| Registered prompt IDs | Worker-local prompt store, capacity 256; avoids reference upload/encoding per turn | IDs disappear on restart or eviction; gateway needs re-registration fallback |
| Token estimate cache | Thread-safe LRU, capacity 4,096 | Saves CPU/tokenization, not GPU generation |
| Output processing | Serial audio-token decode and CPU postprocessing pool | All outputs complete before the batch's futures resolve |
| Sentence splitting | `insert_pauses`: generate sentences sequentially and concatenate WAV plus pauses | Does not overlap sentence generation; pilot was slower for complete responses |
| Lifecycle | Heartbeats, drain, active count, advertised capacity | Advertised capacity is not a hard admission limit; historical `queue_depth` means active requests |
| New GPU ownership fix | Reference encoding, prompt copies, cache operations and generation use the GPU owner thread | Prevents overlap with graph capture; does not increase GPU throughput |
| CUDA failure reporting | Failed contexts refuse work; health becomes 503; callback reports dead | Requires restart for a fatal context; OOM is handled as recoverable |
| New local queue bounds | Limits on pending generations and GPU preparation tasks, HTTP 429, actual queue metrics | Bounds only prepared GPU queues; HTTP thread-pool waiting is still outside them |
| New reference inference mode | `@torch.inference_mode()` on reference helper, executed on GPU owner thread | Avoids building encoder gradients; does not remove the generation latency floor |

The history places initial microbatching in `18ebe97`, further tuning in
`c8e61f1`/`2596aef`, GPU logging in `e31ed17`, VRAM controls in `f1320f3`, and
EC2/cache integration in `d4e7814`. September's `3227f19` adds sentence pauses,
cache-warming/tooling changes and API behavior; it does not add asynchronous
generation waiting, request cancellation or bounded admission. Historical commit
messages claiming throughput are not attributed GPU benchmarks.

Source locations: `src/omnivoice_triton/serving/batching.py`,
`cli/api_server.py`, `models/base_runner.py`, `models/faster_runner.py`,
`models/triton_faster_runner.py`, `serving/worker_lifecycle.py`.

The main remaining ingress bottleneck is synchronous `def /generate`, which
holds a thread while `Future.result()` waits. Starlette's shared default limit
is 40 tokens. Native async waiting can return a complete WAV and requires no
audio-streaming work. Increasing the thread count alone would just move the
queue. [Starlette documentation](https://starlette.dev/threadpool/).

The reference helper's inference mode must execute in the actual worker thread:
PyTorch gradient modes are thread local. Regression tests verify both
`grad_enabled=False` and `inference_mode=True` inside the helper for registration,
cache warming and uploaded clone requests.
[PyTorch 2.8 documentation](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad_mode.inference_mode.html).

## Changes deployed on the 3060

The old worker was drained to zero active requests before its process stopped.
The replacement uses the same model, upstream dependency, identity, public port,
callbacks and protected credentials. It registered successfully and sent healthy
heartbeats to Lingua. MuseTalk was not restarted or modified.

Deployment configuration:

| Setting | Before | Deployed |
| --- | --- | --- |
| Batch collection | 10 ms | 5 ms |
| Maximum batch count | 32 | 8 |
| Target / conditioning token budgets | 24,000 / 12,000 | 4,096 / 4,096 |
| Pending generation / preparation limits | Unbounded | 16 / 4 |
| Graph count cap | 8 | 8 |
| ASR / lazy ASR | Off / off | Off / off |

These are conservative pilot settings, not a proven optimum. HTTP waiting and
single-request maximum work remain future admission work. The onstart launcher
retains these overrides and the qualified feature branch. Automatic pulls into
this unmerged deployment are skipped until review/merge. A root-only rollback
copy of the old launcher, revision,
argv and environment remains on this worker; it must not be copied into reports.
Worker-local prompt IDs were reset by the restart. The Lingua gateway code has
a 404 re-register-and-retry path when its original reference is available. Full
authenticated application playback was not part of this loopback pilot.

## Measurement conclusions

Detailed new 3060 results and raw evidence are in the companion Lingua benchmark
`docs/benchmarks/omnivoice-concurrency-rtx-3060-2026-11-01.md`, with artifacts in
`work/omnivoice-concurrency-2026-11-01/` in the Lingua workspace.

| New RTX 3060 candidate workload | N | p95 complete response |
| --- | --- | --- |
| Cached short clone, 16 steps, loopback | 10 | 440 ms |
| Ten simultaneous short clones, loopback | 10 | 3,818 ms |
| Thirty-word clone, 9.64 seconds of audio, loopback | 3 | 959 ms |
| Same text split and fully rejoined, loopback | 3 | 1,290 ms |
| Short clone over reused public HTTP connection from operator Mac | 10 | 862 ms |

There were 436 loopback and 22 public real-model generation requests including
warmups, six successful reference registrations, and one operator validation
error before inference. The loopback outputs passed finite/nonempty/amplitude
checks; public outputs passed WAV structure checks. Speech accuracy and speaker
similarity were not evaluated, and these small phase samples do not certify
production tail latency. Sixty-one mocked-model regression tests also passed.

The short fixture can produce a complete WAV below 500 ms at low load. Cached
clone generation is slower than voice design. Ten-request bursts and the longer
full responses miss 500 ms. The correctness changes are not evidence of a large
latency speedup. Graph cold starts also miss the target. There is no measured
500 ms capacity for a general production workload yet.

Full-WAV delivery also needs a network budget. At 24 kHz mono PCM16, payload is
48,000 bytes per second of audio. Five thousand 2.2-second responses contain
about 528 MB: moving all payloads in 500 ms needs at least 8.45 Gbit/s aggregate
payload bandwidth, before overhead or preceding generation time. At 9.64 seconds
of audio this rises to about 37 Gbit/s. Avoid copying complete audio through
unnecessary hops, and qualify actual client-to-gateway-to-worker placement.
Operator public-port probes are reported separately from local GPU timing.

Sentence splitting must be judged by **time until every piece is joined**. The
current sequential splitter was slower than whole-text generation in this pilot.
Only measured parallel splitting across spare GPU workers could reduce that
completion time; batching the pieces on the same GPU is not equivalent to having
independent GPUs. Keep text order, voice identity, sample rate and complete lexical
content; evaluate prosody and seams before adopting it.

If the gateway already receives stable LLM sentences, an additional experiment
can overlap internal sentence synthesis with the LLM finishing the rest, then
join all audio before MuseTalk receives anything. This requires no client audio
streaming. Report both full turn latency and time after final text becomes ready;
do not hide early synthesis by moving the timing origin. Pre-generated complete
WAVs for exact common text/voice/settings can reduce cache-hit latency, but report
cache hits separately from new generation and do not extrapolate them to arbitrary
conversation text.

## Implementation sequence

1. **Qualify the complete-response latency floor.** Freeze a production corpus
   by language, voice, 5/15/30/60-word text, actual output duration and settings.
   Test 16 versus 32 steps, cached versus cold voices, full/light/off processing,
   warm versus new graph shapes. Keep the app's video-call profile (16 steps,
   speed 1.0, postprocessing off) distinct from other app defaults. Check word
   completeness, speaker similarity, silence and failures before accepting speed
   changes. Do not truncate a requested utterance to pass a latency target.

2. **Implement async complete-WAV ingress.** Expose nonblocking enqueue/futures,
   use `async def` plus `asyncio.wrap_future`, and await completion without holding
   one HTTP thread per request. Keep heavy CPU preparation in a separate bounded
   executor and every CUDA operation on its owner thread. Make health/drain
   endpoints independent of generation saturation. Bound admission before body
   buffering and expensive preparation by requests, bytes and estimated work.
   Apply the same approach to the blocking Lingua gateway worker hop; changing
   just the worker cannot make a synchronous gateway support 5,000 waiters.

3. **Add deadlines and cancellation.** Propagate one absolute deadline from
   gateway through preparation, queue, generation and response encoding. Expire
   queued work and remove disconnected or superseded call turns. Use future state
   transitions safely so cancellation cannot crash the GPU owner. An in-flight GPU
   kernel is not preempted; suppress stale results and release capacity only when
   actual work ends. Record ingress wait separately from prepared queue wait.
   A timeout or HTTP 429 is not a successful 500 ms generation.

4. **Schedule for full-response deadlines.** Separate short interactive and long
   or cold-preparation work. Use estimated execution cost, aging and per-call
   fairness. Merge only when the measured batch service time fits every request's
   remaining deadline. Benchmark batch sizes 1/2/4/8 and collection windows
   0/2/5/10 ms per GPU; large batches can violate latency even with an empty queue.
   Add an explicit oversized-request policy, sequence-length buckets with correct
   masks and output lengths, measured graph prewarm and a graph VRAM budget.
   Do not simply pad arbitrary audio tokens and assume output parity.

5. **Fix fleet reservations and cache durability.** Atomically select and reserve
   GPU work in Redis (lease + capacity/work limit + expiry in one operation), with
   crash reconciliation. The current router reads load then increments it later;
   these are separate steps. Use queue age and predicted full completion time as
   routing signals. Keep CPU prompt artifacts in a durable, versioned cache and
   warm worker-local copies with voice affinity. Include model/tokenizer revision
   and reference preprocessing in cache identity. Keep cold voice/ASR preparation
   away from latency-sensitive turns. Audit all retry and quota scopes; prevent
   retries from multiplying a synchronized burst.

6. **Choose a faster qualified execution path where needed.** Retest the user's
   five GPU classes against the same current fork and full-output corpus. First
   compare RTX 4070 Super, RTX 3090 and RTX 5080 using the existing 3060 as the
   reference; retain RTX 5060 Ti if its current offer wins the measured cost/SLO
   comparison. These are qualification candidates, not new test results or current
   price claims. Compare base/Triton/hybrid and supported upstream optimizations
   on identical weights/settings. The upstream project has optional acceleration
   paths; its advertised speedups on other GPUs do not establish capacity here.
   [Official OmniVoice repository](https://github.com/k2-fsa/OmniVoice).
   Keep an upstream dependency upgrade separate from scheduler changes, because
   the current deployed lock still uses OmniVoice 0.1.4.

7. **Scale warm capacity through 10 → 100 → 1,000 → 5,000.** At each level run
   the realistic call trace and the synchronized burst independently. Keep reserve
   workers warm, preloaded and in the caller's region. Use queue-age/arrival-rate
   signals and GPU work estimates for autoscaling. Cold provisioning cannot rescue
   a burst that must finish within 500 ms. At 1,000+, partition routing/cache work
   across regional pools; at 5,000, qualify the control plane, network, GPU fleet,
   rolling drain and recovery together. MuseTalk capacity is a separate constraint.

## Capacity math and target stages

Calls are sessions; GPU generations occur only on assistant turns. Start with the
explicit example **one complete assistant response per call every 20 seconds**,
then replace it with application traces. No sentence-to-client streaming is
assumed. If internal splitting uses `k` generation jobs per response, include those
jobs and joins in work estimates rather than treating them as free throughput.

Let `q_slo` be the **measured sustained complete responses/second per GPU**, for
the actual corpus and assigned arrival pattern, while preserving the full network
budget, success rate and p95 target. Include operational headroom in that qualified
value once. Normal-load fleet size is at least `ceil(arrival_rate / q_slo)` plus
failure reserve; validate the resulting fleet trace rather than treating this
average-rate formula as a tail-latency proof.

| Target calls | Example complete-response arrival rate | Synchronized burst to qualify | Stage gate |
| --- | --- | --- | --- |
| 10 | 0.5/s | 10 whole responses | Qualify one warm worker's floor; supply enough independent GPU capacity for burst |
| 100 | 5/s | 100 whole responses | Atomic reservations, cache affinity, bounded admission, N+1 failure test |
| 1,000 | 50/s | 1,000 whole responses | Regional warm pools; async gateway; workload-aware scheduler; sustained soak |
| 5,000 | 250/s | 5,000 whole responses | Sharded control plane, warm burst reserve, network and rolling-restart qualification |

For a burst, separately measure `b500`: whole responses one warm worker can
successfully finish within the **remaining** 500 ms budget. With enough equivalent
independent workers, a lower bound is `ceil(burst / b500)`; validate correlated
delays and routing before certification. For illustration only, if `b500=1`,
10/100/1,000/5,000 simultaneous responses require at least that many warm GPU
workers to serve every response within the deadline. A p95 contract may allow up
to 5% slower results, but should never count rejected requests as compliant.

For normal calls, an illustrative qualified `q_slo=1` would imply
1/5/50/250 serving GPUs for the four arrival rates, before failure reserve.
**This is arithmetic, not a measured sizing recommendation.** Ten short
requests at 1/s on the current 3060 are only a pilot, not sustained mixed-corpus
qualification. For longer responses that already exceed 500 ms on an idle GPU,
replicas alone cannot satisfy the deadline: the service-time floor must fall first.

## Qualification and acceptance

- Generate real speech for bursts of 10/100/1,000/5,000 only on an adequately
  provisioned isolated fleet. Load-test async admission/control-plane behavior
  separately with a mocked GPU backend; label those results explicitly.
- Use both closed-loop call sessions with measured think/speech intervals and
  open-loop arrivals with scheduled start times. Report scheduler/client lateness
  to avoid hiding overload. Add correlated bursts, cache misses, mixed languages,
  long replies, unique voices, worker loss and cancellations.
- Measure request-send → complete WAV received → ready for MuseTalk separately.
  Log p50/p95/p99, all-attempt deadline success, valid WAV and speech content,
  rejection/timeout/empty/silent rates, actual queue age, graph captures, allocated
  and reserved VRAM, device memory, utilization and cost per qualifying response.
- Proposed budget: 50–100 ms for network/gateway/encoding, leaving 400–450 ms
  for preparation, queue and full generation. Measure joint end-to-end tails;
  separate stage p95s do not simply add into an end-to-end p95 guarantee.
- Suggested acceptance for the agreed corpus: p95 complete response <500 ms,
  >=99.9% valid completed responses, no lexical omissions beyond the baseline
  tolerance, no growing queue or memory trend, and a 30–60 minute sustained run
  plus worker-loss recovery. These are proposed gates, not measured results.
- Keep the existing 4070 Ti failure-parity findings separate: short Arabic
  silence/postprocessing failures persisted in upstream and optimized paths.
  Fast HTTP 200 with silent audio is not successful TTS; bound retries inside
  the original deadline and include their extra GPU work.

## Practical verdict

Keep the complete-WAV → MuseTalk contract. The current fork has meaningful
microbatching and graph/cache work, but it cannot meet a universal 500 ms
complete-output promise at 10–5,000 simultaneous requests. The existing 3060 is
useful for low-rate short responses and correctness work. The next implementation
priority is async bounded admission, deadline-aware scheduling and atomic fleet
reservations, followed by GPU qualification on the actual full-response corpus.
