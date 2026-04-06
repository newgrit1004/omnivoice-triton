# Benchmark Results

Detailed benchmark results for omnivoice-triton on RTX 5090.

## Hardware & Environment

| Item | Spec |
|------|------|
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120, 32GB) |
| CUDA | 12.8 |
| PyTorch | 2.8 (cu128) |
| Triton | bundled with PyTorch 2.8 |
| Model | OmniVoice (Qwen3-0.6B backbone, 28 layers) |
| OS | WSL2 (Linux 5.15) |
| Python | 3.12 |
| dtype | bfloat16 |
| Batch Size | 1 |

---

## Kernel Micro-benchmarks

Measured with `triton.testing.do_bench()`, batch=1, seq_len=512, hidden=1024.

| Kernel | PyTorch (ms) | Triton (ms) | Speedup | HBM Savings |
|--------|:------------:|:-----------:|:-------:|:-----------:|
| RMSNorm | 0.02665 | 0.00452 | **5.90x** | 4→1 round-trips |
| SwiGLU | 0.00996 | 0.00696 | **1.43x** | 3→1 round-trips |
| FusedAddRMSNorm | 0.02875 | 0.00643 | **4.47x** | 2 kernels → 1 |

**Notes:**
- RMSNorm fuses variance accumulation, normalization, and weight scaling into a single SRAM pass.
- SwiGLU fuses `silu(gate) * up`, eliminating the intermediate tensor allocation.
- FusedAddRMSNorm fuses the residual addition and RMSNorm into one kernel, reducing HBM round-trips from 2 separate kernels to 1.
- Run `make bench-kernels` to reproduce on your hardware.

---

## E2E Inference Benchmarks

Measured with `torch.cuda.Event` timing. 3 warmup runs, 5 measured runs per text.
RTF = audio\_duration / generation\_time. Higher RTF means faster-than-real-time.

### Korean — "안녕하세요, 오늘 날씨가 정말 좋네요."

| Runner | Mean (ms) | Std (ms) | Min (ms) | p50 (ms) | p95 (ms) | RTF (mean) | Peak VRAM |
|--------|:---------:|:--------:|:--------:|:--------:|:--------:|:----------:|:---------:|
| Base | 556.2 | 55.4 | 501.4 | 519.3 | 634.3 | 4.74x | 1.945 GB |
| Triton | 519.5 | 35.7 | 484.3 | 505.3 | 572.9 | 5.01x | 1.946 GB |
| Faster | 192.7 | 32.5 | 166.2 | 166.7 | 236.8 | 14.58x | 1.965 GB |
| **Hybrid** | **165.2** | 36.0 | 131.0 | 146.6 | 211.6 | **16.63x** | 1.988 GB |

### English — "Hello, welcome to the OmniVoice text-to-speech system."

| Runner | Mean (ms) | Std (ms) | Min (ms) | p50 (ms) | p95 (ms) | RTF (mean) | Peak VRAM |
|--------|:---------:|:--------:|:--------:|:--------:|:--------:|:----------:|:---------:|
| Base | 780.5 | 165.5 | 543.3 | 852.4 | 955.6 | 4.56x | 1.955 GB |
| Triton | 512.2 | 42.6 | 440.2 | 515.8 | 563.3 | 6.51x | 1.956 GB |
| Faster | 226.7 | 37.3 | 180.0 | 250.8 | 263.4 | 15.99x | 1.986 GB |
| **Hybrid** | **179.2** | 36.8 | 145.5 | 154.5 | 226.1 | **20.44x** | 2.010 GB |

### Chinese — "你好，今天天气真好。"

| Runner | Mean (ms) | Std (ms) | Min (ms) | p50 (ms) | p95 (ms) | RTF (mean) | Peak VRAM |
|--------|:---------:|:--------:|:--------:|:--------:|:--------:|:----------:|:---------:|
| Base | 572.8 | 42.7 | 506.9 | 567.0 | 629.1 | 3.53x | 1.932 GB |
| Triton | 506.1 | 16.4 | 489.0 | 501.4 | 530.5 | 3.95x | 1.932 GB |
| Faster | 187.7 | 36.4 | 156.5 | 160.1 | 234.1 | 11.70x | 1.972 GB |
| **Hybrid** | **164.2** | 44.2 | 121.2 | 149.0 | 226.8 | **13.81x** | 1.997 GB |

---

## Summary Table

| Runner | Korean | English | Chinese | Avg Speedup | Peak VRAM |
|--------|:------:|:-------:|:-------:|:-----------:|:---------:|
| Base | 556ms | 781ms | 573ms | 1.00x | 1.95 GB |
| Triton | 519ms | 512ms | 506ms | 1.02x | 1.95 GB |
| Faster | 193ms | 227ms | 188ms | 2.75x | 1.98 GB |
| **Hybrid** | **165ms** | **179ms** | **164ms** | **3.26x** | 2.00 GB |

---

## RTF Analysis

RTF (Real-Time Factor) measures how many seconds of audio are generated per second of compute. RTF > 1.0 means faster-than-real-time.

| Runner | RTF (Korean) | RTF (English) | RTF (Chinese) | Avg RTF |
|--------|:-----------:|:------------:|:------------:|:-------:|
| Base | 4.74x | 4.56x | 3.53x | 4.28x |
| Triton | 5.01x | 6.51x | 3.95x | 5.16x |
| Faster | 14.58x | 15.99x | 11.70x | 14.09x |
| **Hybrid** | **16.63x** | **20.44x** | **13.81x** | **16.96x** |

Hybrid mode generates audio at roughly 17x real-time on average — a 3.26x improvement over the PyTorch baseline.

---

## Model Load Times

| Runner | Load Time (s) |
|--------|:------------:|
| Base | 4.40 |
| Triton | 2.80 |
| Faster | 2.76 |
| Hybrid | 2.82 |

Triton/Faster/Hybrid load faster than Base because kernel compilation is cached after the first run.

---

## Methodology Notes

- **Timing**: `torch.cuda.Event` with `synchronize()` before and after each generation.
- **Warmup**: 3 runs discarded to stabilize CUDA Graph replay and JIT compilation.
- **Repetitions**: 5 measured runs per text per runner.
- **VRAM**: `torch.cuda.max_memory_allocated()` reset before each timed run.
- **Text selection**: Short utterances (10–20 tokens) representative of typical TTS workloads.
- **No batching**: All measurements at batch_size=1 (online inference scenario).

> **Disclaimer**: Benchmarks measured on a single RTX 5090. Results vary with GPU model, driver version, system load, and input text length. Run `make bench-speed` on your hardware for accurate numbers.
