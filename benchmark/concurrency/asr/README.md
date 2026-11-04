GPU: rented NVIDIA GeForce RTX 3090, physical 24,576 MiB VRAM, driver 580.65.06,
verified with nvidia-smi. PyTorch-visible VRAM: 25,296,044,032 bytes. Isolated ASR
runtime: Torch 2.8.0+cu128 / CUDA 12.8. Per-run metadata records the date and
co-resident processes. Local syntax/synthetic scoring checks were CPU only on
Apple M5 Pro; targeted Whisper checks used that Mac's MLX/Metal runtime.

Actual GPU ASR completed on November 3, 2026 (America/Chicago; November 4 UTC),
after TTS timing, in two passes. Installed Omnilingual enumerated 1,672 supported
language/script IDs; this experiment tested only English, Spanish and Arabic.

| Pass | Logical saved-audio cases per judge | Unique WAV transcription calls per judge | ASR input segments per judge | HTTP200 audio missing |
|---|---:|---:|---:|---:|
| Initial | 980 | 938 | 974 | 0 |
| Corrected follow-up | 127 | 109 | 119 | 0 |

Each pass includes three original reference controls; repeated exact hashes reuse
transcripts within a pass. Segments are ASR-only <=30-second chunks, not additional
independent quality samples. Across both passes, all 1,053 successful TTS WAVs were
screened; 18,134 HTTP429 responses are separate overload rejections. Codec fixtures
and reference controls are additional quality cases, not HTTP successes.

Corrected startup-seeded sequential before/cache32: 18/18 WAV SHA256 and waveform
sample pairs are identical, including three warmups and fifteen text cases. Both
judges therefore have identical paired scores. This supports no regression in
those fixtures; existing baseline quality issues remain. Original metadata-only
seed experiments and the mislabeled first grouped HTTP experiment are retained
with explicit protocol warnings. The actual corrected grouped HTTP profile is
reported separately. No blanket quality or 100-language qualification is claimed.

These are offline quality judges for complete saved OmniVoice WAVs. ASR must run
after TTS timing. Both variants may use the same rented GPU sequentially; isolate
their Python environment and release TTS first if VRAM is insufficient. Running
ASR concurrently with timed TTS would contaminate latency and capacity results.

Selected judges:

| Model card | Architecture | Upstream BF16 inference estimate | FP32 download |
|---|---|---|---|
| `omniASR_LLM_Unlimited_1B_v2` | autoregressive decoder, known language/script | ~6 GiB | 8.5 GiB |
| `omniASR_CTC_1B_v2` | parallel CTC, no language conditioning | ~3 GiB | 3.7 GiB |

These are **Meta's A100 / batch1 / 30-second audio estimates**, not measured VRAM
or performance on the rented GPU. Unlimited LLM supports long inputs, but we feed
both judges the same <=30-second ASR-only chunks for comparable conditions. For
longer WAVs, the script finds a quiet boundary between 22 and 29 seconds; boundary
errors remain possible and are flagged. Original complete WAVs are never edited.

Official code/models are Apache-2.0. Official dataset is separately CC-BY-4.0.
This harness does not download the dataset or call a transcription API.

Install in an isolated Linux x86_64 Python3.12 environment:

```bash
bash install.sh /workspace/omnivoice-speedups/asr-venv
```

This pins official `omnilingual-asr` source at commit
`81f51e224ce9e74b02cc2a3eaf21b2d91d743455`, `fairseq2==0.6`, matching C++ extension
`fairseq2n==0.6+cu128`, `torch==2.8.0+cu128`, `torchaudio==2.8.0+cu128`,
`pyarrow==20.0.0`, `soundfile==0.13.1`, `jiwer==4.0.0`. Arrow 20 is pinned because
newer Arrow 26 requires NumPy 2 while fairseq2 0.6 requires NumPy 1.26. Exact transitive versions are written to
`installed-packages.txt`. System libsndfile1 is required. CUDA driver compatibility
must be verified on the rental before inference. PyPI 0.2.0 declares Python<=3.12,
which excludes Python 3.12 patch releases; the pinned official source explicitly
supports >=3.10,<3.14. We use the reviewed source commit rather than suppressing
Python compatibility checks or installing an older model library.

Input is a JSON list or JSONL with unique `id`, WAV `path` (relative to manifest),
`sha256`, `expected`/`text`, and either `language` (en/es/ar) or exact Meta
`lang_id`/`omnilingual_language`. Other languages require explicit script IDs,
checked against installed `supported_langs`. Expected text is used only after
transcription; it is never passed as a prompt or context to either ASR model.

```bash
/workspace/omnivoice-speedups/asr-venv/bin/python evaluate.py \
  --manifest /workspace/run/baseline/quality-manifest.json \
  --output /workspace/run/baseline/omnilingual
/workspace/omnivoice-speedups/asr-venv/bin/python capture_assets.py \
  --output /workspace/run/baseline/omnilingual/assets-sha256.json
```

Each model runs in its own subprocess, releasing its GPU allocation on exit. All
cases from one model finish before the next model starts. Outputs retain original
WAV hashes, per-model transcripts, normalized WER/CER, S/D/I word alignment, exact
match, ASR wall time, GPU/driver/runtime/package metadata, and review flags. Hash
the fairseq2 cache after first use because model-card URLs are not immutable
revision IDs. Capture assets and package freeze alongside saved before/after WAVs.
Within one model run, byte-identical original WAV hashes with the same language
reuse the first transcript. Every logical case remains in results, reuse is
explicitly tagged, and summaries report unique audio hashes and actual inference
counts. Repeated copies of identical audio are not independent quality evidence.

Run reference recordings through both judges as calibration controls before
attributing synthesized-audio mismatches to OmniVoice. Use identical texts,
voices, steps and audio profiles for paired before/after evaluation. The current
HTTP API exposes no per-request seed. Concurrent/unseeded samples are stochastic.
A separate startup wrapper can seed Torch/CUDA before module import and record
validated server evidence; matched sequential runs can then be hash-compared.
The corrected fixture above produced identical WAVs under that controlled setup.

```bash
/workspace/omnivoice-speedups/asr-venv/bin/python compare.py \
  --baseline /workspace/run/baseline/omnilingual/scored.json \
  --candidate /workspace/run/candidate/omnilingual/scored.json \
  --output /workspace/run/quality-comparison
```

This emits `comparison.json` and a `review.md` with direct before/after WAV links,
per-judge transcript/edit deltas, missing pairs, and per-language review counts.
Missing outputs must be joined to request/HTTP logs; do not silently exclude
rejections from availability denominators or treat them as quality passes.

For the full rental experiment, `run_all.py` discovers every retained successful
audio manifest under `results/`, adds the three unchanged reference controls,
captures HTTP/audio coverage separately, and runs both judges once across all
arms. Prefixing IDs preserves logical rows while allowing exact-hash transcript
reuse across arms. It then splits per-arm scores and writes paired comparisons.
Token-only scoring probes are excluded because their check is token equality.

```bash
# CPU-only file/coverage preparation, no model load:
/workspace/omnivoice-speedups/asr-venv/bin/python run_all.py --prepare-only
# GPU inference: run only AFTER all TTS measurements/probes finish:
/workspace/omnivoice-speedups/asr-venv/bin/python run_all.py
```

The API returns text, not a calibrated confidence score. `confidence` is therefore
null. Flags represent ASR disagreement, text mismatch, chunk boundary exposure or
clipping, not numerical confidence or automatically proven TTS errors. Any mismatch
requires review; two judges agreeing is still no perceptual quality certification.
Both variants share the Omnilingual family. WER is not comparable across writing
systems, so CER and language-aware review remain necessary. Do not accept a speed
change based only on aggregate WER; inspect paired outputs, all deletions, repeats,
pronunciation, voice consistency, and listening reviews.

Observed ASR limitations are recorded without changing strict scores: English
`to day` versus `today` creates nonzero WER with CER0; written digits/currency can
disagree with spelled-number references. Both Omnilingual variants omitted Arabic
numbers in one retained clip that offline Whisper recognized. Separate targeted
Whisper checks of 12 purposively selected clips are stored locally, not used to
estimate a global failure rate. Review flags are not confirmed TTS failures.

Sources checked 2026-11-03:

- [Official models, memory table, installation and license](https://github.com/facebookresearch/omnilingual-asr)
- [Official inference reference](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/README.md)
- [Versioned model cards](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/cards/models/rc_models_v2.yaml)
- [fairseq2 ABI installation requirements](https://github.com/facebookresearch/fairseq2#installing-on-linux)
