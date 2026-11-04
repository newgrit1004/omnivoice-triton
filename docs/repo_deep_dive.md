# OmniVoice-Triton Repository Deep Dive

## Scope

This document audits the **117 pre-existing tracked files** in this repository as returned by `git ls-files` at the start of the audit.

- It covers source code, tests, docs, workflows, benchmark artifacts, UI assets, and committed sample WAVs.
- It does **not** cover `.git/`, `.venv/`, or other untracked/generated local files.
- The most important boundary: this repository is **not the full OmniVoice model implementation**. It is an **optimization/runtime wrapper** around the installed upstream `omnivoice` package.

In practical terms:

- `src/omnivoice_triton/...` owns the **Triton kernels**, **runner classes**, **monkey-patching**, **CUDA Graph capture**, **benchmarks**, and **demo UI**.
- The installed upstream `omnivoice` package owns the **core TTS model**, including token preparation, duration estimation, iterative unmasking, audio token decoding, and audio post-processing.

That boundary matters because to explain “how TTS is created” end to end, you need both sides of the call chain:

1. This repo chooses and optimizes the execution mode.
2. Upstream `omnivoice` actually performs the text-to-audio-token generation and waveform decoding.

## What The TTS Pipeline Actually Does

### 1. Runner selection

User code enters through:

- `omnivoice_triton.create_runner("base" | "triton" | "faster" | "hybrid")`

That factory lives in `src/omnivoice_triton/models/__init__.py` and selects one of four runner classes:

- `BaseRunner`: plain OmniVoice inference
- `TritonRunner`: OmniVoice plus fused Triton kernels
- `FasterRunner`: OmniVoice plus CUDA Graph replay
- `TritonFasterRunner`: Triton patching first, then CUDA Graph capture

Two additional “modes” appear elsewhere in the repo:

- `triton_sage`
- `hybrid_sage`

These are not standalone runner names in `create_runner()`. They are implemented by passing `enable_sage_attention=True` into the `triton` or `hybrid` runners.

### 2. Model loading

All runners ultimately rely on `BaseRunner.load_model()`, which calls:

```python
OmniVoice.from_pretrained(
    self.model_id,
    device_map=self.device,
    dtype=self.dtype,
)
```

The upstream `omnivoice` package then loads:

- the OmniVoice wrapper model
- the internal Qwen3-style LLM backbone
- a text tokenizer
- a Higgs audio tokenizer/decoder
- a feature extractor
- a rule-based duration estimator

The upstream model structure is:

- text input is tokenized by a text tokenizer
- audio is represented as **8 codebooks**
- each codebook uses a vocabulary of **1025 IDs**
- the LLM hidden states are projected by `audio_heads` into `8 x 1025` logits per time step

### 3. Mode-specific optimization

#### Base

Nothing is patched. Every forward pass uses normal PyTorch/Transformers modules.

#### Triton

`TritonRunner.load_model()` finds the patchable model and applies three kinds of kernel replacement:

- RMSNorm modules are replaced by `TritonRMSNorm`
- MLP forward methods are replaced to use fused `silu(gate) * up`
- decoder-layer forward methods are replaced so residual add plus post-attention RMSNorm become one fused kernel

By default, the runner uses `patch_range=(0, 24)`, so the first 24 decoder layers are patched and the last 4 remain in vanilla PyTorch.

#### Faster

`FasterRunner` wraps `model.forward` in `_CUDAGraphForward`.

The first time a given input shape is seen:

1. static input buffers are cloned
2. a warmup forward pass runs on a side stream
3. the forward pass is captured into `torch.cuda.CUDAGraph`

On later calls with the same shape:

1. new inputs are copied into the static buffers
2. `graph.replay()` is called
3. the static output buffer is returned

This works unusually well for OmniVoice because it is **non-autoregressive** and repeatedly refines a fixed-length target sequence, so tensor shapes remain stable across its iterative decoding steps.

#### Hybrid

`TritonFasterRunner` applies Triton patches **before** installing the CUDA Graph wrapper. That means the captured graph already contains the fused kernels.

### 4. Generation API

The public runner methods are:

- `generate(text, language=None, ...)`
- `generate_voice_clone(text, ref_audio, ref_text="", ...)`
- `generate_voice_design(text, instruct, ...)`

Each runner method:

1. checks that the model is loaded
2. resets CUDA peak-memory stats
3. constructs `OmniVoiceGenerationConfig`
4. forwards the request to `self._model.generate(...)`
5. converts the returned waveform to NumPy
6. returns a dict with audio, sample rate, elapsed time, and peak VRAM

### 5. Upstream OmniVoice preprocessing

Inside the upstream `omnivoice` package, generation goes through these phases:

#### Text-only / “default voice”

- text is normalized into a batch
- language is resolved to a supported code if provided
- no reference audio is used
- duration is estimated using a rule-based character/script weighting heuristic

If there is no reference text/audio, the estimator falls back to a built-in heuristic reference:

- text: `"Nice to meet you."`
- token count: `25`

#### Voice clone

Reference audio is converted into a reusable `VoiceClonePrompt`:

1. audio is loaded and resampled to the tokenizer sampling rate
2. stereo is downmixed to mono
3. RMS is measured
4. optionally long audio is trimmed
5. silence is removed from the reference
6. if transcript is missing, Whisper ASR can auto-transcribe it
7. audio is encoded into 8-codebook audio tokens with the upstream audio tokenizer

That prompt contributes:

- `ref_audio_tokens`
- `ref_text`
- `ref_rms`

The target duration is then estimated by comparing target text weight against reference text weight and reference audio-token length.

#### Voice design

The `instruct` string is normalized through upstream voice-design utilities:

- mutually exclusive attributes are validated
- English/Chinese instruct tags are normalized
- language/style tags are inserted into the prompt

No reference audio is used.

### 6. Input packing for the model

Upstream `_prepare_inference_inputs()` builds a single packed sequence containing:

1. style tokens
   - optional `<|denoise|>`
   - `<|lang_start|>...<|lang_end|>`
   - `<|instruct_start|>...<|instruct_end|>`
2. wrapped text tokens
   - `<|text_start|> ... <|text_end|>`
3. optional reference audio tokens
4. a fully masked target-audio region

The target-audio region is initialized with the audio mask token ID `1024` across all 8 codebooks.

### 7. Iterative unmasking

This is the real TTS decoding loop.

For each item, OmniVoice:

1. estimates the number of target audio tokens
2. creates a schedule for how many masked positions to fill at each step
3. duplicates the batch into:
   - conditional rows
   - unconditional rows
4. runs the model repeatedly for `num_step` iterations, default `32`
5. applies classifier-free guidance:
   - conditional logits
   - unconditional logits
   - guidance scale
6. picks candidate token IDs and confidence scores
7. optionally perturbs position selection with Gumbel noise
8. unmasks the top-`k` most confident still-masked positions
9. writes those tokens back into the packed sequence

This is why the repo’s CUDA Graph optimization is effective:

- the same model is run many times
- with identical layout and near-identical shapes
- on the same fixed-length target region

### 8. Waveform decoding and post-processing

When all 8-codebook tokens are filled:

1. the upstream audio tokenizer decodes tokens into waveform chunks
2. chunked outputs are cross-faded if long text was split
3. optional silence removal is applied to the generated output
4. volume is scaled against the reference RMS for voice cloning
5. otherwise the waveform is peak-normalized
6. fade-in/out and silence padding are added to avoid clicks

The final product returned to the user is a 24 kHz waveform array.

## Repository Layout Summary

| Area | Files | Role |
|---|---:|---|
| Root/project metadata | 12 | packaging, docs, licensing, repo automation |
| `src/omnivoice_triton/` | 13 | runtime package, kernels, runners |
| `benchmark/` | 10 | perf and verification tooling plus stored results |
| `scripts/` | 2 | asset/document generation helpers |
| `tests/` | 7 | kernel tests and end-to-end sanity tests |
| `ui/` | 15 | Streamlit dashboard and translations |
| `assets/audio_samples/` | 56 | committed demo audio plus metadata |
| `docs/` | 2 | benchmark narratives |

## File-By-File Deep Dive

### Root Files

#### `.gitignore`

- Ignores Python build/cache artifacts, virtualenvs, coverage output, and IDE clutter.
- Also ignores several project-specific files that the public README refers to, including `Makefile`.
- Important implication: the README’s `make ...` commands depend on a file that is **not tracked in this repo**.

#### `.gitmessage.txt`

- Conventional commit template with `feat/fix/docs/...` style guidance.
- Has no runtime effect; it standardizes commit authoring.

#### `.pre-commit-config.yaml`

- Configures:
  - `pre-commit-hooks` for JSON/YAML/TOML/EOF/whitespace checks
  - `ruff` linting and formatting
  - local `ty` type checking via `uv run ty check`
- This governs contributor hygiene, not model behavior.

#### `CITATION.cff`

- Defines formal citation metadata for the package.
- This is documentation/distribution metadata, not code execution.

#### `LICENSE`

- Apache 2.0 license.
- Governs redistribution and modification rights.

#### `README.md`

- Main English project narrative.
- Explains optimization goals, install flow, runner modes, benchmarks, verification strategy, and sample assets.
- Describes the project as an OmniVoice acceleration layer, not a fresh TTS model.
- Contains some stale or inconsistent claims versus code:
  - references make targets without a tracked `Makefile`
  - benchmark numbers vary between sections
  - some text implies all 28 layers are patched, but default runner settings patch only layers `0..23`

#### `README_ko.md`

- Korean-language counterpart to the README.
- Shares the same high-level story: NAR iterative unmasking plus Triton/CUDA Graph acceleration.
- Not a byte-for-byte mirror; some numbers and wording differ from the English README.

#### `pyproject.toml`

- The central packaging and tool configuration file.
- Defines:
  - package metadata (`name`, `version`, license, Python range)
  - runtime dependencies (`triton`, `transformers`, `omnivoice`, `sageattention`, etc.)
  - optional extras (`dev`, `eval`, `ui`, `all`)
  - Hatch build settings
  - UV index routing for CUDA 12.8 PyTorch wheels
  - Ruff, pytest, ty, and coverage settings
- Runtime significance:
  - `omnivoice` is a required dependency, confirming that the full TTS core is external to this repo
  - `sageattention` is in core dependencies, so Sage support is intended to be first-class
  - pytest default config skips `tests/test_model_parity.py` unless called directly

#### `uv.lock`

- UV’s generated lockfile with exact dependency resolution.
- Ensures reproducible environments, including the pinned versions of `omnivoice`, `transformers`, `torch`, `streamlit`, and eval dependencies.
- This file is deployment-critical but not algorithmically interesting; it does not change how TTS works, only which dependency versions implement it.

### GitHub Automation

#### `.github/dependabot.yml`

- Schedules weekly updates for:
  - GitHub Actions dependencies
  - pip dependencies declared at repo root
- Keeps automation and package pins current.

#### `.github/workflows/ci.yml`

- CI pipeline for pushes and pull requests to `main`.
- Installs UV and Python 3.12, syncs dependencies, then runs:
  - `ruff check`
  - `ruff format --check`
  - `ty check`
  - `pytest -m "not gpu"`
- Important limitation: GPU kernels are not tested in CI here, only CPU-safe coverage.

#### `.github/workflows/publish.yml`

- Tag-triggered release pipeline.
- Verifies that the Git tag version matches `pyproject.toml`.
- Builds a wheel/sdist and smoke-tests importability before publishing to PyPI or TestPyPI.
- Smoke test only imports `__version__`; it does not validate GPU/TTS execution.

### Python Package: `src/omnivoice_triton/`

#### `src/omnivoice_triton/__init__.py`

- Public package entry point.
- Defines `__version__ = "0.1.0"`.
- Runs `_check_torch()` at import time:
  - raises if PyTorch is missing
  - warns if CUDA is unavailable
- Re-exports all public kernels and runner helpers.
- Effectively makes `import omnivoice_triton` a runtime environment check plus API surface export.

#### `src/omnivoice_triton/py.typed`

- Empty PEP 561 marker file.
- Signals that the package ships inline type information.

#### `src/omnivoice_triton/kernels/__init__.py`

- Re-export layer for kernel classes/functions.
- Establishes the package’s three core optimized ops as public API.

#### `src/omnivoice_triton/kernels/utils.py`

- Defines `calculate_settings(n)`.
- Uses Triton heuristics to map hidden-size width to:
  - `BLOCK_SIZE = next_power_of_2(n)`
  - `num_warps` tuned by block size
- This is shared launch-configuration logic for all three kernels.

#### `src/omnivoice_triton/kernels/rms_norm.py`

- Implements fused RMSNorm in Triton.
- Kernel formula:

  ```text
  y = weight * x / sqrt(mean(x^2) + eps)
  ```

- Key implementation details:
  - flattens input to `(N, H)` regardless of original leading dimensions
  - loads one row of activations and weights from HBM
  - computes variance in fp32 (“llama casting mode”)
  - casts normalized activations back to original dtype before weight multiply
  - stores output in a single pass
- Exposes:
  - `_rms_norm_forward_kernel`
  - `triton_rms_norm(x, weight, eps)`
  - `TritonRMSNorm(nn.Module)`
- This is one of the main bandwidth-saving wins in the repo.

#### `src/omnivoice_triton/kernels/swiglu.py`

- Implements fused SwiGLU:

  ```text
  out = silu(gate) * up
  ```

- Key implementation details:
  - validates shape equality and CUDA placement
  - reshapes to 2-D row-major form
  - computes sigmoid in fp32 for stability
  - avoids materializing an intermediate `silu(gate)` tensor in PyTorch eager mode
- Exposes:
  - `_swiglu_forward_kernel`
  - `triton_swiglu_forward(gate, up)`
  - `TritonSwiGLU(nn.Module)`

#### `src/omnivoice_triton/kernels/fused_norm_residual.py`

- Implements fused residual add plus RMSNorm.
- Kernel formula:

  ```text
  s = x + residual
  y = weight * s / sqrt(mean(s^2) + eps)
  ```

- Key implementation details:
  - performs residual add in fp32
  - writes back both:
    - normalized output
    - updated residual tensor
  - reuses the fp32 intermediate for variance computation
- Exposes:
  - `_fused_add_rms_norm_forward_kernel`
  - `triton_fused_add_rms_norm(x, residual, weight, eps)`
  - `TritonFusedAddRMSNorm(nn.Module)`
- This is specifically designed to replace the “residual add, then post-attention layer norm” pattern in decoder layers.

#### `src/omnivoice_triton/models/__init__.py`

- Re-export and factory module for runners.
- Defines `_RUNNER_MAP`, `ALL_RUNNER_NAMES`, `get_runner_class()`, and `create_runner()`.
- Only supports the four canonical names:
  - `base`
  - `triton`
  - `faster`
  - `hybrid`

#### `src/omnivoice_triton/models/base_runner.py`

- Minimal runner wrapper around upstream OmniVoice.
- Core responsibilities:
  - map dtype strings to `torch.dtype`
  - load the model
  - forward generation requests
  - measure elapsed time and peak VRAM
  - convert returned audio to NumPy
- `load_model()` is the handoff point into the true TTS engine.
- `generate()`, `generate_voice_clone()`, and `generate_voice_design()` differ only in which kwargs they forward to `OmniVoice.generate()`.
- This file does not implement synthesis logic itself; it provides a stable, benchmark-friendly API around it.

#### `src/omnivoice_triton/models/faster_runner.py`

- Adds CUDA Graph replay to `BaseRunner`.
- `_CUDAGraphForward` stores a graph entry per `input_ids.shape`.
- `_capture()`:
  - clones static buffers for every dynamic tensor input
  - runs one warmup forward pass
  - captures the forward pass in a CUDA Graph
- `__call__()`:
  - falls back to original forward if training or labels are present
  - copies new tensors into static buffers
  - replays the graph
- `FasterRunner.load_model()` installs this wrapper by replacing `self._model.forward`.
- This is where most of the repo’s end-to-end speedup comes from.

#### `src/omnivoice_triton/models/patching.py`

- The most important orchestration file in the package.
- It owns all monkey-patching logic.

- `_get_layer_index(name)`:
  extracts decoder-layer indices from dotted module names.

- `_should_patch(name, patch_range)`:
  gates patching so only the intended layer range is modified.

- `_get_parent(model, dotted_name)`:
  resolves a module path so a child module can be replaced in-place.

- `_replace_rms_norm(...)`:
  swaps an existing RMSNorm module for `TritonRMSNorm` while reusing the original weight parameter.

- `_patch_mlp_forward(mlp)`:
  replaces the module’s `forward()` with:

  ```python
  gate = self.gate_proj(x)
  up = self.up_proj(x)
  return self.down_proj(triton_swiglu_forward(gate, up))
  ```

- `_patch_decoder_layer_forward(layer)`:
  replaces the entire decoder-layer forward path so the residual add and post-attention normalization are fused.
  It dynamically supports either:
  - `post_attention_layernorm`
  - `post_self_attn_layernorm`

- `_detect_sage_kernel()` and `_get_sage_kernel()`:
  choose the best SageAttention CUDA kernel for the active GPU architecture.

- `_patch_attention_sage(attn)`:
  replaces attention forward with SageAttention when:
  - no attention mask is present
  - no KV cache is in use
  Otherwise it falls back to PyTorch SDPA.

- `apply_sage_attention(model, patch_range=None)`:
  patches all matching attention modules.

- `apply_triton_kernels(model, enable_fused_norm=True, patch_range=None)`:
  patches RMSNorms, MLPs, and optionally decoder-layer fused norm/residual logic.

- `find_patchable_model(model)`:
  finds the inner `nn.Module` to patch.

This file is the bridge between generic upstream model code and optimized runtime behavior.

#### `src/omnivoice_triton/models/triton_runner.py`

- Extends `BaseRunner`.
- After model load, it:
  - finds the patchable model
  - applies Triton kernel patches
  - optionally applies SageAttention
- Default `patch_range=(0, 24)` keeps the last 4 layers unpatched.

#### `src/omnivoice_triton/models/triton_faster_runner.py`

- Extends `FasterRunner`.
- Deliberately avoids `FasterRunner.load_model()` so it can patch first.
- Actual order:
  1. call `BaseRunner.load_model(self)`
  2. patch Triton kernels
  3. optionally patch SageAttention
  4. install `_CUDAGraphForward`
- This order is essential because graph capture must see the already-optimized kernels.

### Benchmarks: `benchmark/`

#### `benchmark/bench_kernels.py`

- Micro-benchmark harness for the three kernel replacements.
- Uses OmniVoice-relevant tensor sizes:
  - hidden size `1024`
  - intermediate size `3072`
- Defines PyTorch reference implementations for:
  - RMSNorm
  - SwiGLU
  - fused add+RMSNorm
- Measures runtime with `triton.testing.do_bench()`.
- Writes results to `benchmark/results/kernel_benchmarks.json`.

#### `benchmark/bench_e2e.py`

- End-to-end benchmark harness across six configurations:
  - Base
  - Triton
  - Triton+Sage
  - Faster
  - Hybrid
  - Hybrid+Sage
- Measures:
  - model load time
  - mean/std/p50/p95 latency
  - RTF
  - peak VRAM
- Uses CUDA events for accurate GPU timing.
- Saves `benchmark/results/e2e_benchmarks.json`.

#### `benchmark/bench_voice_clone.py`

- Specialized benchmark for voice cloning.
- Searches the Hugging Face cache for LJSpeech WAV files and uses them as reference audio.
- Benchmarks only the four canonical runner names from `ALL_RUNNER_NAMES`.
- Saves `voice_clone_benchmarks.json` if run, but that artifact is not tracked in this repo.

#### `benchmark/eval_config.py`

- Central config for Tier 3 quality evaluation.
- Defines:
  - evaluation sentences in Korean, English, and Chinese
  - thresholds for UTMOS, CER, speaker similarity, and deltas vs baseline
  - runner comparison pairs
  - voice-design instruct string
- Some values are currently informational more than authoritative because parts of the UI import this file incorrectly.

#### `benchmark/eval_quality.py`

- Tier 3 quality-evaluation pipeline.
- Generates audio independently for the baseline and optimized runners, then computes:
  - CER via Whisper + `jiwer`
  - UTMOS via `SpeechMOS`
  - speaker similarity via `resemblyzer` or MFCC fallback
- Supports `fast` and `full` modes.
- Builds per-runner distributions and compares them statistically.
- Writes `tier3_{mode}_multi.json`.
- This file is the repo’s richest quality-audit logic.

#### `benchmark/run_verification.py`

- Orchestrator for the repo’s “3-tier verification” story.
- Tier 1:
  runs kernel pytest tests.
- Tier 2:
  runs `tests/test_model_parity.py`.
- Tier 3:
  shells out to `eval_quality.py`.
- Writes `benchmark/results/verification_report.json`.
- Important nuance:
  it records Tier 3 success based on **subprocess exit code**, not on the logical PASS/FAIL verdict inside `tier3_fast_multi.json`.

#### `benchmark/results/kernel_benchmarks.json`

- Stored micro-benchmark artifact with three rows:
  - RMSNorm
  - SwiGLU
  - FusedAddRMSNorm
- Shows the committed kernel-level performance evidence.

#### `benchmark/results/e2e_benchmarks.json`

- Stored end-to-end benchmark artifact.
- Contains per-language benchmark rows for:
  - Base
  - Triton
  - Triton+Sage
  - Faster
  - Hybrid
  - Hybrid+Sage
- This is the raw data source used by the UI and docs.

#### `benchmark/results/tier3_fast_multi.json`

- Stored detailed fast-mode quality-evaluation artifact.
- Contains:
  - ref metrics for `base`
  - opt metrics for each optimized mode
  - per-sentence samples and transcripts
  - comparison verdicts
- The committed artifact currently marks all optimized configs as `FAIL` because their CER delta exceeds the configured threshold, even though speaker similarity stays very high.

#### `benchmark/results/verification_report.json`

- Stored verification summary artifact written by `run_verification.py`.
- Current contents show:
  - Tier 1: `PASS`, `60/60`
  - Tier 2: `PASS`, but `tests=[]` and `total=0`
  - Tier 3: `PASS` because the evaluation subprocess exited cleanly
- This file is therefore useful as a run log, but not a perfect reflection of logical verification quality.

### Scripts: `scripts/`

#### `scripts/generate_bench_tables.py`

- Reads benchmark JSON files and formats Markdown tables.
- Intended to update README sections between marker comments.
- Important implementation detail:
  - script expects `<!-- BENCH:KERNELS:START -->`
  - README uses `<!-- BENCH:KERNEL:START -->`
- Because of that marker mismatch, kernel-table injection into the README will not work automatically.

#### `scripts/generate_samples.py`

- Generates committed demo audio samples and `assets/audio_samples/metadata.json`.
- Defines:
  - per-language sample texts
  - optional voice-design sample configs
  - six output modes, including Sage variants
- For each mode:
  - creates the correct runner
  - loads the model
  - synthesizes each sample
  - saves a WAV
  - records generation metadata
- Important caveat:
  the voice-design branch calls `runner.generate(text=..., instruct=...)` instead of `runner.generate_voice_design(...)`.
  That means voice-design sample generation is currently broken and silently skipped under the script’s `try/except`.
- That bug matches the current committed metadata, which contains only standard generate-mode samples.

### Tests: `tests/`

#### `tests/__init__.py`

- Empty package marker.

#### `tests/kernels/__init__.py`

- Empty subpackage marker.

#### `tests/kernels/conftest.py`

- Provides fixtures:
  - `hidden_size() -> 2048`
  - `eps() -> 1e-6`
  - `intermediate_size() -> 6144`
- These fixture sizes do not match the main OmniVoice benchmark sizes elsewhere in the repo and appear to be leftover or generic helpers.

#### `tests/kernels/test_rms_norm.py`

- GPU-only correctness tests for the fused RMSNorm kernel.
- Compares Triton output against a PyTorch reference implementation using fp32 variance accumulation.
- Covers:
  - multiple shapes
  - fp16 and bf16
  - 2-D inputs
  - non-contiguous inputs
  - float32 mode
  - single-token edge case
  - determinism
  - numerical stability

#### `tests/kernels/test_swiglu.py`

- GPU-only correctness tests for fused SwiGLU.
- Checks:
  - shape/dtype parity vs `F.silu(gate) * up`
  - module wrapper behavior
  - error handling for mismatched shape and CPU tensors
  - 2-D inputs
  - non-contiguous inputs
  - determinism
  - stability under extreme values

#### `tests/kernels/test_fused_norm.py`

- GPU-only correctness tests for fused residual add + RMSNorm.
- Validates both:
  - updated residual tensor
  - normalized output tensor
- Covers fp16/bf16, module wrapper, 2-D inputs, non-contiguous inputs, determinism, and stability.

#### `tests/test_model_parity.py`

- Slow GPU integration test over the four main runner modes.
- Despite the filename, it does **not** compare layer-level logits or cosine similarity.
- It only checks that generated audio is:
  - non-empty
  - non-silent
  - within a reasonable duration range
- This makes it an output-sanity smoke test, not a true parity test.

### UI Package: `ui/`

#### `ui/__init__.py`

- Package marker with a short module docstring.

#### `ui/app.py`

- Streamlit entry point.
- Adds repo root to `sys.path`, then creates five tabs:
  - Overview
  - Inference
  - Benchmarks
  - Audio Samples
  - Verification

#### `ui/charts.py`

- Plotly rendering helpers for:
  - kernel latency bars
  - kernel speedup bars
  - E2E latency by runner/language
  - RTF by runner/language
  - verification layer cosine charts
- Gracefully degrades with `st.warning()` when Plotly is not installed.

#### `ui/gpu_info.py`

- Runtime GPU-inspection helper.
- Prefers `pynvml` for:
  - model name
  - driver version
  - total/used/free VRAM
  - utilization
  - temperature
- Falls back to `torch.cuda` if NVML is unavailable.

#### `ui/i18n.py`

- Lightweight singleton translation manager.
- Loads all locale JSON files from `ui/locales/`.
- Exposes:
  - `SUPPORTED_UI_LANGS`
  - `I18n.get()`
  - `t(key, **kwargs)`
- Falls back to English or the raw key if a translation is missing.

#### `ui/locales/en.json`

- English source translation map with **129 keys**.
- Covers app title, tab names, benchmark labels, inference controls, verification labels, and table headers.

#### `ui/locales/ko.json`

- Korean translation map with the same **129-key** schema.
- Localizes the full UI surface.

#### `ui/locales/zh.json`

- Chinese translation map with the same **129-key** schema.
- Localizes the full UI surface.

#### `ui/sidebar.py`

- Renders:
  - project title/subtitle
  - UI language selector
  - optional GPU quick-info card
  - project links
- Uses `get_gpu_info()` and `I18n`.

#### `ui/utils.py`

- Shared utility functions for the dashboard.
- Provides:
  - current/peak VRAM queries
  - VRAM reset
  - RTF calculation
  - speedup formatting
  - JSON list/dict loading with error handling

#### `ui/tab_overview.py`

- Renders the landing overview tab.
- Shows:
  - project summary
  - runner cards
  - kernel cards
  - GPU info
  - verification summary
  - a quick benchmark table
- Important detail:
  some overview values are hardcoded and stale, such as “4 runners” and “~2.8x” max speedup.

#### `ui/tab_inference.py`

- Interactive TTS tab.
- Supports:
  - standard TTS
  - voice cloning
  - voice design
- Runner choice is limited to the four canonical modes; Sage variants are not exposed here.
- Voice cloning uploads reference audio to a temp file, then calls `generate_voice_clone`.
- Voice design correctly calls `generate_voice_design`.

#### `ui/tab_benchmarks.py`

- Reads benchmark JSON artifacts and displays them as tables plus charts.
- Uses `e2e_benchmarks.json` and `kernel_benchmarks.json`.
- Generic enough to display Sage rows if present in the underlying data.

#### `ui/tab_samples.py`

- Displays committed audio samples side by side.
- Can fall back to raw directory scanning if metadata is missing.
- Important caveats:
  - `_MODES` includes only `base`, `triton`, `faster`, `hybrid`, so committed `triton_sage` and `hybrid_sage` samples are ignored in the side-by-side UI.
  - `_GEN_TYPE_LABELS` expects `basic/clone/design`, but metadata uses `generate`, so sample-type labeling is not aligned with current artifacts.

#### `ui/tab_verification.py`

- Renders verification status cards and detailed tier sections.
- Has several schema assumptions that do not match current stored artifacts:
  - expects rich Tier 2 pair/layer data, but `verification_report.json` stores only pytest summary counts
  - tries to import `EVAL_CONFIG` from `benchmark.eval_config`, but the file defines `EVAL_CONFIGS`
  - reads `tier3_fast_multi.json` for detailed comparisons, which is more accurate than `verification_report.json`

### Docs: `docs/`

#### `docs/benchmark_results_en.md`

- Static English benchmark report.
- Summarizes kernel and E2E results using curated tables and narrative notes.
- Focuses on the four main modes and omits Sage rows from the summary tables.

#### `docs/benchmark_results_ko.md`

- Korean version of the benchmark report.
- Mirrors the English doc structurally.

### Assets: `assets/audio_samples/`

#### `assets/audio_samples/README.md`

- Describes the sample directory layout and how to regenerate samples.
- Claims “All audio is saved as 32-bit float WAV at 24 kHz.”
- That claim does **not** match the committed files, which are currently **16-bit PCM mono WAVs**.

#### `assets/audio_samples/metadata.json`

- Central manifest for the committed audio sample set.
- Records:
  - generation timestamp
  - hardware info
  - modes
  - per-sample metadata
- Current metadata contains **54 samples**:
  - 6 modes
  - 3 languages
  - 3 standard text samples per language
- It does **not** contain voice-design or voice-clone outputs.

#### `assets/audio_samples/base/en_00.wav`
#### `assets/audio_samples/base/en_01.wav`
#### `assets/audio_samples/base/en_02.wav`
#### `assets/audio_samples/base/ko_00.wav`
#### `assets/audio_samples/base/ko_01.wav`
#### `assets/audio_samples/base/ko_02.wav`
#### `assets/audio_samples/base/zh_00.wav`
#### `assets/audio_samples/base/zh_01.wav`
#### `assets/audio_samples/base/zh_02.wav`
#### `assets/audio_samples/triton/en_00.wav`
#### `assets/audio_samples/triton/en_01.wav`
#### `assets/audio_samples/triton/en_02.wav`
#### `assets/audio_samples/triton/ko_00.wav`
#### `assets/audio_samples/triton/ko_01.wav`
#### `assets/audio_samples/triton/ko_02.wav`
#### `assets/audio_samples/triton/zh_00.wav`
#### `assets/audio_samples/triton/zh_01.wav`
#### `assets/audio_samples/triton/zh_02.wav`
#### `assets/audio_samples/triton_sage/en_00.wav`
#### `assets/audio_samples/triton_sage/en_01.wav`
#### `assets/audio_samples/triton_sage/en_02.wav`
#### `assets/audio_samples/triton_sage/ko_00.wav`
#### `assets/audio_samples/triton_sage/ko_01.wav`
#### `assets/audio_samples/triton_sage/ko_02.wav`
#### `assets/audio_samples/triton_sage/zh_00.wav`
#### `assets/audio_samples/triton_sage/zh_01.wav`
#### `assets/audio_samples/triton_sage/zh_02.wav`
#### `assets/audio_samples/faster/en_00.wav`
#### `assets/audio_samples/faster/en_01.wav`
#### `assets/audio_samples/faster/en_02.wav`
#### `assets/audio_samples/faster/ko_00.wav`
#### `assets/audio_samples/faster/ko_01.wav`
#### `assets/audio_samples/faster/ko_02.wav`
#### `assets/audio_samples/faster/zh_00.wav`
#### `assets/audio_samples/faster/zh_01.wav`
#### `assets/audio_samples/faster/zh_02.wav`
#### `assets/audio_samples/hybrid/en_00.wav`
#### `assets/audio_samples/hybrid/en_01.wav`
#### `assets/audio_samples/hybrid/en_02.wav`
#### `assets/audio_samples/hybrid/ko_00.wav`
#### `assets/audio_samples/hybrid/ko_01.wav`
#### `assets/audio_samples/hybrid/ko_02.wav`
#### `assets/audio_samples/hybrid/zh_00.wav`
#### `assets/audio_samples/hybrid/zh_01.wav`
#### `assets/audio_samples/hybrid/zh_02.wav`
#### `assets/audio_samples/hybrid_sage/en_00.wav`
#### `assets/audio_samples/hybrid_sage/en_01.wav`
#### `assets/audio_samples/hybrid_sage/en_02.wav`
#### `assets/audio_samples/hybrid_sage/ko_00.wav`
#### `assets/audio_samples/hybrid_sage/ko_01.wav`
#### `assets/audio_samples/hybrid_sage/ko_02.wav`
#### `assets/audio_samples/hybrid_sage/zh_00.wav`
#### `assets/audio_samples/hybrid_sage/zh_01.wav`
#### `assets/audio_samples/hybrid_sage/zh_02.wav`

- These committed WAVs are **evidence artifacts**, not source code.
- They demonstrate the audible output of each runner mode for the same short utterances.
- All committed sample WAVs inspected here are:
  - mono
  - 24,000 Hz
  - 16-bit PCM
- Their exact per-file metadata appears in the appendix table below.

## Known Caveats And Inconsistencies Found During The Audit

1. The repo accelerates OmniVoice, but the actual TTS model implementation is external in the installed `omnivoice` package, not vendored here.
2. The README and docs rely on `make` commands, but there is no tracked `Makefile`; `.gitignore` explicitly ignores it.
3. Default Triton and Hybrid runners patch only layers `0..23`, not all 28 decoder layers.
4. `scripts/generate_samples.py` calls the wrong API for voice design, so voice-design sample generation is effectively broken.
5. `assets/audio_samples/README.md` says samples are 32-bit float WAVs, but the committed files are 16-bit PCM.
6. `ui/tab_samples.py` ignores committed Sage sample directories and expects sample types that do not match current metadata.
7. `scripts/generate_bench_tables.py` and `README.md` disagree on kernel marker names, so automatic kernel table injection will fail.
8. `tests/test_model_parity.py` is an output-validity smoke test, not a true model-parity check despite its name.
9. `benchmark/results/verification_report.json` can report Tier 3 `PASS` even when `tier3_fast_multi.json` contains logical comparison failures, because the wrapper only checks subprocess exit status.
10. `ui/tab_verification.py` expects richer Tier 2 and config schemas than the current stored artifacts actually provide.

## Appendix: Committed Audio Sample Inventory

| File | Mode | Lang | Audio Duration | Generation Time | Sample Rate | Channels | Encoding | Text |
|---|---|---|---:|---:|---:|---:|---|---|
| `base/ko_00.wav` | base | ko | 1.66s | 1.081s | 24000 | 1 | 16-bit | 안녕하세요, 반갑습니다. |
| `base/ko_01.wav` | base | ko | 1.79s | 0.554s | 24000 | 1 | 16-bit | 오늘 날씨가 정말 좋네요. |
| `base/ko_02.wav` | base | ko | 2.35s | 0.609s | 24000 | 1 | 16-bit | 옴니보이스 음성 합성 시스템입니다. |
| `base/en_00.wav` | base | en | 1.96s | 0.578s | 24000 | 1 | 16-bit | Hello, nice to meet you. |
| `base/en_01.wav` | base | en | 1.90s | 0.466s | 24000 | 1 | 16-bit | The weather is really nice today. |
| `base/en_02.wav` | base | en | 2.71s | 0.579s | 24000 | 1 | 16-bit | Welcome to the OmniVoice text-to-speech system. |
| `base/zh_00.wav` | base | zh | 1.66s | 0.563s | 24000 | 1 | 16-bit | 你好，很高兴认识你。 |
| `base/zh_01.wav` | base | zh | 1.82s | 0.480s | 24000 | 1 | 16-bit | 今天天气真好。 |
| `base/zh_02.wav` | base | zh | 2.70s | 0.571s | 24000 | 1 | 16-bit | 欢迎使用OmniVoice语音合成系统。 |
| `triton/ko_00.wav` | triton | ko | 1.67s | 0.726s | 24000 | 1 | 16-bit | 안녕하세요, 반갑습니다. |
| `triton/ko_01.wav` | triton | ko | 1.77s | 0.485s | 24000 | 1 | 16-bit | 오늘 날씨가 정말 좋네요. |
| `triton/ko_02.wav` | triton | ko | 2.35s | 0.450s | 24000 | 1 | 16-bit | 옴니보이스 음성 합성 시스템입니다. |
| `triton/en_00.wav` | triton | en | 1.96s | 0.499s | 24000 | 1 | 16-bit | Hello, nice to meet you. |
| `triton/en_01.wav` | triton | en | 1.88s | 0.422s | 24000 | 1 | 16-bit | The weather is really nice today. |
| `triton/en_02.wav` | triton | en | 2.70s | 0.479s | 24000 | 1 | 16-bit | Welcome to the OmniVoice text-to-speech system. |
| `triton/zh_00.wav` | triton | zh | 1.67s | 0.495s | 24000 | 1 | 16-bit | 你好，很高兴认识你。 |
| `triton/zh_01.wav` | triton | zh | 1.82s | 0.459s | 24000 | 1 | 16-bit | 今天天气真好。 |
| `triton/zh_02.wav` | triton | zh | 2.72s | 0.556s | 24000 | 1 | 16-bit | 欢迎使用OmniVoice语音合成系统。 |
| `triton_sage/ko_00.wav` | triton_sage | ko | 1.67s | 0.522s | 24000 | 1 | 16-bit | 안녕하세요, 반갑습니다. |
| `triton_sage/ko_01.wav` | triton_sage | ko | 1.77s | 0.456s | 24000 | 1 | 16-bit | 오늘 날씨가 정말 좋네요. |
| `triton_sage/ko_02.wav` | triton_sage | ko | 2.35s | 0.565s | 24000 | 1 | 16-bit | 옴니보이스 음성 합성 시스템입니다. |
| `triton_sage/en_00.wav` | triton_sage | en | 1.96s | 0.512s | 24000 | 1 | 16-bit | Hello, nice to meet you. |
| `triton_sage/en_01.wav` | triton_sage | en | 1.88s | 0.484s | 24000 | 1 | 16-bit | The weather is really nice today. |
| `triton_sage/en_02.wav` | triton_sage | en | 2.70s | 0.532s | 24000 | 1 | 16-bit | Welcome to the OmniVoice text-to-speech system. |
| `triton_sage/zh_00.wav` | triton_sage | zh | 1.67s | 0.535s | 24000 | 1 | 16-bit | 你好，很高兴认识你。 |
| `triton_sage/zh_01.wav` | triton_sage | zh | 1.82s | 0.419s | 24000 | 1 | 16-bit | 今天天气真好。 |
| `triton_sage/zh_02.wav` | triton_sage | zh | 2.72s | 0.523s | 24000 | 1 | 16-bit | 欢迎使用OmniVoice语音合成系统。 |
| `faster/ko_00.wav` | faster | ko | 1.66s | 0.334s | 24000 | 1 | 16-bit | 안녕하세요, 반갑습니다. |
| `faster/ko_01.wav` | faster | ko | 1.79s | 0.293s | 24000 | 1 | 16-bit | 오늘 날씨가 정말 좋네요. |
| `faster/ko_02.wav` | faster | ko | 2.35s | 0.426s | 24000 | 1 | 16-bit | 옴니보이스 음성 합성 시스템입니다. |
| `faster/en_00.wav` | faster | en | 1.96s | 0.388s | 24000 | 1 | 16-bit | Hello, nice to meet you. |
| `faster/en_01.wav` | faster | en | 1.90s | 0.389s | 24000 | 1 | 16-bit | The weather is really nice today. |
| `faster/en_02.wav` | faster | en | 2.71s | 0.254s | 24000 | 1 | 16-bit | Welcome to the OmniVoice text-to-speech system. |
| `faster/zh_00.wav` | faster | zh | 1.66s | 0.386s | 24000 | 1 | 16-bit | 你好，很高兴认识你。 |
| `faster/zh_01.wav` | faster | zh | 1.82s | 0.385s | 24000 | 1 | 16-bit | 今天天气真好。 |
| `faster/zh_02.wav` | faster | zh | 2.70s | 0.424s | 24000 | 1 | 16-bit | 欢迎使用OmniVoice语音合成系统。 |
| `hybrid/ko_00.wav` | hybrid | ko | 1.67s | 0.286s | 24000 | 1 | 16-bit | 안녕하세요, 반갑습니다. |
| `hybrid/ko_01.wav` | hybrid | ko | 1.77s | 0.336s | 24000 | 1 | 16-bit | 오늘 날씨가 정말 좋네요. |
| `hybrid/ko_02.wav` | hybrid | ko | 2.35s | 0.305s | 24000 | 1 | 16-bit | 옴니보이스 음성 합성 시스템입니다. |
| `hybrid/en_00.wav` | hybrid | en | 1.96s | 0.258s | 24000 | 1 | 16-bit | Hello, nice to meet you. |
| `hybrid/en_01.wav` | hybrid | en | 1.88s | 0.332s | 24000 | 1 | 16-bit | The weather is really nice today. |
| `hybrid/en_02.wav` | hybrid | en | 2.70s | 0.154s | 24000 | 1 | 16-bit | Welcome to the OmniVoice text-to-speech system. |
| `hybrid/zh_00.wav` | hybrid | zh | 1.67s | 0.270s | 24000 | 1 | 16-bit | 你好，很高兴认识你。 |
| `hybrid/zh_01.wav` | hybrid | zh | 1.82s | 0.360s | 24000 | 1 | 16-bit | 今天天气真好。 |
| `hybrid/zh_02.wav` | hybrid | zh | 2.72s | 0.294s | 24000 | 1 | 16-bit | 欢迎使用OmniVoice语音合成系统。 |
| `hybrid_sage/ko_00.wav` | hybrid_sage | ko | 1.67s | 0.275s | 24000 | 1 | 16-bit | 안녕하세요, 반갑습니다. |
| `hybrid_sage/ko_01.wav` | hybrid_sage | ko | 1.77s | 0.330s | 24000 | 1 | 16-bit | 오늘 날씨가 정말 좋네요. |
| `hybrid_sage/ko_02.wav` | hybrid_sage | ko | 2.35s | 0.260s | 24000 | 1 | 16-bit | 옴니보이스 음성 합성 시스템입니다. |
| `hybrid_sage/en_00.wav` | hybrid_sage | en | 1.96s | 0.356s | 24000 | 1 | 16-bit | Hello, nice to meet you. |
| `hybrid_sage/en_01.wav` | hybrid_sage | en | 1.88s | 0.273s | 24000 | 1 | 16-bit | The weather is really nice today. |
| `hybrid_sage/en_02.wav` | hybrid_sage | en | 2.70s | 0.127s | 24000 | 1 | 16-bit | Welcome to the OmniVoice text-to-speech system. |
| `hybrid_sage/zh_00.wav` | hybrid_sage | zh | 1.67s | 0.320s | 24000 | 1 | 16-bit | 你好，很高兴认识你。 |
| `hybrid_sage/zh_01.wav` | hybrid_sage | zh | 1.82s | 0.245s | 24000 | 1 | 16-bit | 今天天气真好。 |
| `hybrid_sage/zh_02.wav` | hybrid_sage | zh | 2.72s | 0.270s | 24000 | 1 | 16-bit | 欢迎使用OmniVoice语音合成系统。 |
