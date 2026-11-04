#!/usr/bin/env bash
# Linux x86_64 / Python3.12 / CUDA12.8 judge-only environment.
set -euo pipefail
asr_venv="${1:-/workspace/omnivoice-speedups/asr-venv}"
python312="${ASR_PYTHON:-python3.12}"
"$python312" -m venv "$asr_venv"
"$asr_venv/bin/python" -m pip install --upgrade pip
"$asr_venv/bin/python" -m pip install 'torch==2.8.0+cu128' 'torchaudio==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
# The C++ extension must match PyTorch and CUDA exactly; avoid PyPI CPU wheel.
"$asr_venv/bin/python" -m pip install 'fairseq2n==0.6+cu128' --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
# PyPI0.2.0 incorrectly declares <=3.12, excluding Python3.12 patch releases.
# Current official source supports >=3.10,<3.14; pin that exact reviewed commit.
"$asr_venv/bin/python" -m pip install 'fairseq2[arrow]==0.6' 'omnilingual-asr @ git+https://github.com/facebookresearch/omnilingual-asr.git@81f51e224ce9e74b02cc2a3eaf21b2d91d743455' 'pyarrow==20.0.0' 'soundfile==0.13.1' 'jiwer==4.0.0'
"$asr_venv/bin/python" -m pip check
"$asr_venv/bin/python" -m pip freeze > "$asr_venv/installed-packages.txt"
"$asr_venv/bin/python" - <<'PY'
import torch
import fairseq2
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available(), 'languages': len(supported_langs)})
assert torch.cuda.is_available(), 'CUDA is required for this GPU qualification run'
assert all(x in supported_langs for x in ['eng_Latn', 'spa_Latn', 'arb_Arab'])
PY
