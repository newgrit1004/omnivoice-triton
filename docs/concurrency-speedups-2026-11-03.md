# Concurrency speedups: measured RTX 3090 pilot

**GPU: NVIDIA GeForce RTX 3090, physical VRAM 24,576 MiB (`nvidia-smi`), visible VRAM 25,296,044,032 bytes (PyTorch).** Driver 580.65.06; CUDA 12.8; Torch 2.8.0+cu128; Triton 3.4.0; Transformers 5.5.4; OmniVoice 0.1.4; Python 3.12.3. Date: November 3, 2026, America/Chicago (November 4 UTC). CPU: AMD Ryzen 9 5950X, with 32 visible logical CPUs. This was an isolated rented Vast worker; the existing RTX 3060 and MuseTalk were not changed.

The measured improvements are responsive HTTP/control handling and lower latency for a repeated mixed workload with a larger graph cache on this 24 GB card. Async waiting alone did not materially speed GPU inference. The 32-graph profile improved the later staggered mixed phase, while novel mixed bursts remained slow. Complete-WAV p95 below 500 ms was not achieved across workloads.

## Code and repeatable protocol

- `a026a89`: await GPU futures without occupying shared HTTP worker threads; use separate bounded CPU/prompt preparation pools and safely cancel queued jobs.
- `0b75539`: add optional grouped codec decoding for equal-length tokens. The default batch size of 1 retains serial decoding. Batch size 8 remains experimental and off by default because its strict numerical screen failed.
- `5558b22`: stop emitting full exception tracebacks for expected overload rejections. HTTP 429 and inference behavior are preserved.
- The 32-graph cache is a launch-profile change, not a new universal default. Maximum sampled device memory was 12.32 GiB, versus 10.16 GiB before. This profile was evaluated only on the RTX 3090.

The baseline (`0e94f2c`) already includes the user fork's CUDA owner-thread/race fixes, bounded GPU queues and inference-mode clone encoding. The model checkpoint is `k2-fsa/OmniVoice`, revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.

The [benchmark protocol](../benchmark/concurrency/bench/README.md) and [local ASR setup](../benchmark/concurrency/asr/README.md) retain reproducible inputs, model pins, package/model hashes and validation. The main profile uses hybrid FP16, no SageAttention, 16 steps, speed 1.0, denoise on, postprocessing off and one complete WAV per response. Batching uses a 5 ms collection window, up to 8 requests, target/conditioning token budgets of 4,096/4,096, 16 pending generation slots and 4 pending GPU tasks. No client audio streaming is used.

## Complete-WAV latency

**GPU for every row: RTX 3090, 24,576 MiB physical VRAM / 25,296,044,032 bytes visible.** Each primary arm has 170 attempts: 88 HTTP 200 responses and 82 HTTP 429 rejections. Table p95 values use HTTP 200 responses only. Rejections are not successful concurrency capacity.

| Phase | HTTP 200 / attempts per arm | Before p95 (ms) | Async p95 (ms) | Cache 32 p95 (ms) | Experimental codec 8 p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-short-repeat10 | 10/10 | 512.9 | 517.4 | 442.1 | 524.3 |
| burst10-short | 10/10 | 3,901.9 | 3,941.0 | 3,741.0 | 3,998.5 |
| burst10-mixed | 10/10 | 12,106.9 | 12,235.1 | 12,680.3 | 12,010.4 |
| burst100-short | 18/100 | 7,315.2 | 7,374.7 | 7,205.8 | 7,464.1 |
| staggered-mixed-0.5rps | 18/18 | 2,835.7 | 2,847.3 | 2,307.8 | 2,854.1 |

The mixed staggered phase median changed from 1,041.7 to 704.0 ms, and p95 changed from 2,835.7 to 2,307.8 ms, about 18.6% lower. The 8-graph cache captured 51 times for 21 unique shapes, with 43 evictions; the 32-graph cache captured 21 times, with no evictions. This is an observed profile comparison with finite, unseeded samples, not an isolated causal or universal speedup claim. It favors this repeated shape corpus; unrestricted production text and voices can still churn the cache.

In the cache-32 short sequential phase, all 10 responses completed within 500 ms, with p95 about 442 ms. Before optimization, 9/10 met that limit, with p95 about 513 ms. Ten measurements are a pilot, not SLO certification. ASR setup/download CPU and I/O activity overlapped some initial timing; ASR GPU inference never overlapped timed TTS. Model stage timers measure host dispatch and can charge asynchronous GPU work to later stages; the isolated codec probe uses synchronization and CUDA events.

With 128 pending slots, all 100 requests completed in both baseline and async variants. Baseline health probes had four timeouts at the 5-second client limit; async had none, with maximum completed health latency about 175 ms. Generation p95 remained about 36.6–37.0 seconds. Cancellation removes an explicitly cancelled queued task; automatic cancellation for disconnected calls and request deadlines remain future work.

The real HTTP overload pilots offered 1,000 and 5,000 requests with 16 pending slots. Before optimization, 26/1,000 and 58/5,000 completed; async completed 34/1,000 and 66/5,000. Remaining requests received HTTP 429. These runs include client dispatch and new-connection effects, including a roughly 2.1–2.3-second dispatch spread for 5,000 requests, and accepted request sets differ. They demonstrate bounded overload behavior, not support for 1,000 or 5,000 successful simultaneous calls. A warm fleet, gateway admission/routing and measured per-GPU SLO capacity remain required.

Loopback metrics exclude the external network, Lingua gateway, MuseTalk and playback. Reference recordings are 12–15 seconds, deliberately longer than the usual 3–10-second guidance. Three reference voices were registered before timed generation, and requests reused their prompt IDs. Reference-registration latency is excluded; cold new-voice requests are not qualified by the warm short-response result.

## Audio preservation and word checks

**All 18 corrected startup-seeded before/cache-32 pairs are byte-identical WAVs with bit-identical decoded samples.** These include 3 warmups and 15 English, Spanish and Arabic texts, including negations, names and spoken numbers. Both ASR judges therefore produced zero paired score changes. See the [15 saved corpus pairs](../benchmark/concurrency/samples/rtx3090-2026-11-03/README.md) and their [hashes, texts and profile](../benchmark/concurrency/samples/rtx3090-2026-11-03/manifest.json). Startup seed `20261103` is applied before package/model import and validated in `server-seed.json`; the API has no per-request seed, and concurrent scheduling is not paired.

The open-source local GPU judges are `omniASR_LLM_Unlimited_1B_v2` and `omniASR_CTC_1B_v2`, both under Apache-2.0. The installation enumerated 1,672 language/script IDs. The isolated ASR environment uses Torch 2.8+cu128, fairseq2 0.6 and pinned official Omnilingual commit `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`. ASR ran sequentially on the same RTX 3090 after TTS timing. Expected text is never supplied as an ASR prompt. No paid transcription API was called.

Two ASR passes covered 980 and 127 logical WAV records, containing 938 and 109 unique hashes within their respective passes. These include controls, codec fixtures and recordings from retained invalid-protocol runs. Every one of the 1,053 saved HTTP 200 API WAVs was matched to both judges. These quality checks covered **English, Spanish and Arabic** only, not all 1,672 language/script IDs.

| RTX 3090 load arm | WAVs | CTC aggregate WER | LLM aggregate WER |
| --- | ---: | ---: | ---: |
| before | 88 | 3.51% | 6.68% |
| async-only | 88 | 2.06% | 6.26% |
| cache32 | 88 | 2.89% | 7.91% |
| grouped-decode8-corrected | 88 | 3.03% | 7.02% |

These unseeded aggregate scores do not establish quality equivalence or regression. Strict WER flags ASR spacing (`to day`), spelling and numbers/currency formatting. Both Omnilingual variants can share errors: both omitted numbers from one Arabic recording where a blind offline Whisper cross-check heard them. CTC recognized all 3 original controls exactly; the LLM judge repeated a final Arabic word in 1 control. An English omission and an unclear Arabic short recording remain substantive review flags. All 18 seeded pairs preserve the same existing baseline flags; this proves preservation in those fixtures, not universal speech correctness or perceptual quality.

Additional blind Whisper checks ran on the operator's **Apple M5 Pro with 24 GiB unified memory, using MLX/Metal**, separate from RTX 3090 TTS/ASR inference. These are targeted checks, not independent overall failure-rate estimates. No human MOS or speaker-similarity study was performed.

The isolated codec component experiment used 3 reference recordings with 8 copies each. Median time was 618.18 ms for serial decoding versus 540.78 ms for grouped decoding, about 12.5% less component time, measured with synchronization. The worst maximum sample difference, 0.001918, exceeds the predeclared `1e-4` limit; minimum SNR was 64.37 dB. Both judges recognized all 24 logical recordings per codec arm exactly, representing 3 unique recordings per arm. The flag stays off by default; these results do not establish an end-to-end speed gain.

## Validation and experimental decisions

- 71 fake-model API/batcher/codec tests passed with CUDA visible on the RTX 3090. These are separate from real-model inference measurements. Static checks and a CPU-only seeded-launcher RNG probe passed.
- A deterministic batched-scoring prototype matched predictions, confidence scores and final tokens bitwise for 9 English/Spanish/Arabic cases at batch sizes 1, 2 and 8. Measured timing changes were below 1%; no production scoring patch was added.
- A harness audit caught options recorded but not forwarded. Original `grouped-decode8` runs actually used serial decoding with batch size 1; original `*-seeded-critical` runs were unseeded. All originals remain retained and excluded from intended-profile conclusions. Fresh corrected runs check actual health/seed evidence before generation. The verifier accepts 11 valid arms and rejects exactly 3 original invalid-protocol arms.
- Generation steps, playback speed and audio length were not reduced to produce the measured changes. Grouped codec decoding remains opt-in; the 32-graph profile remains specific to the tested hardware. Future optimization acceptance requires matched inputs, complete saved audio, evidence that runtime options were applied, quality review, and separate latency/availability denominators.

Next work includes request deadlines and disconnect cancellation, sequence bucketing with memory budgets, an async gateway with atomic fleet reservation, and language-specific ASR calibration and listening review. Realistic calls and synchronized bursts at 10, 100, 1,000 and 5,000 calls still require separate fleet qualification.
