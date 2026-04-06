# Bug Fix Report: CUDA Graph Buffer Corruption & SageAttention Kernel Selection

> Date: 2026-04-06
> Affected files: `src/omnivoice_triton/models/faster_runner.py`, `src/omnivoice_triton/models/patching.py`

---

## Summary

3-Tier 품질 평가(Tier 3) 과정에서 2개의 심각한 버그를 발견하고 수정했습니다.

| Bug | 영향 | 심각도 |
|-----|------|--------|
| CUDA Graph static buffer 누락 | Faster/Hybrid 러너가 가비지 오디오 생성 | **Critical** |
| SageAttention 잘못된 API 사용 | Sage 변형 러너의 UTMOS 1.25 (사용 불가 수준) | **Critical** |

---

## Bug 1: CUDA Graph — attention_mask Static Buffer 누락

### 문제

CUDA Graph capture 시 `input_ids`와 `audio_mask`만 static buffer로 관리하고, **`attention_mask`는 일반 텐서로 전달**했습니다. CUDA Graph는 capture 시점의 GPU 메모리 주소를 기록하므로, 해당 텐서의 메모리가 해제되거나 다른 용도로 재사용되면 replay 시 **가비지 데이터를 읽습니다**.

### 증상

```
Base [ko]:    "인공지능 음성 합성 기술이 빠르게 발전하고 있습니다."  ← 정확
Faster [ko]:  "Yeah, we're having a good week."                      ← 가비지
Faster [zh]:  "HAHAHAHA"                                             ← 가비지
Hybrid [zh]:  "Luo Bai road"                                         ← 가비지
```

- 한국어 입력 → 영어 출력, 중국어 입력 → 독일어 출력 등 **완전히 다른 언어의 오디오** 생성
- RMS(에너지)는 정상 범위(0.04~0.09)이므로 단순 에너지 검사로는 탐지 불가
- Whisper ASR + CER 검증으로만 발견 가능

### 원인 분석

OmniVoice의 forward() 호출 인터페이스:
```python
model.forward(input_ids, audio_mask, labels, attention_mask, document_ids, position_ids)
```

Iterative unmasking (32회 반복) 중 매 호출마다 전달되는 인자:
| 인자 | Shape 예시 | 값 변화 | Static Buffer |
|------|-----------|---------|:---:|
| `input_ids` | (2, 8, 131) | 매 반복 변경 (토큰 언마스킹) | **있음** |
| `audio_mask` | (2, 131) | 고정 | **있음** |
| `attention_mask` | **(2, 1, 131, 131)** | **고정** | **없음 (버그)** |
| `document_ids` | None | N/A | N/A |
| `position_ids` | None | N/A | N/A |

`attention_mask`는 shape이 고정이고 값도 변하지 않지만, **CUDA Graph capture 시 GPU 주소만 기록**됩니다. 해당 텐서 메모리가 PyTorch CUDA allocator에 의해 회수되면 replay 시 임의의 데이터를 읽게 됩니다.

### 수정 내용

**파일**: `src/omnivoice_triton/models/faster_runner.py`

**Before** (버그):
```python
def _capture(self, input_ids, audio_mask, **kwargs):
    static_input_ids = input_ids.clone()
    static_audio_mask = audio_mask.clone()
    # kwargs(attention_mask 등)는 그대로 전달 — GPU 주소만 기록됨
    ...
    with torch.cuda.graph(graph):
        static_output = self._original_forward(
            static_input_ids, static_audio_mask, **kwargs
        )

def __call__(self, input_ids, audio_mask, ...):
    ...
    entry["static_input_ids"].copy_(input_ids)
    entry["static_audio_mask"].copy_(audio_mask)
    # attention_mask는 업데이트 안 됨!
    entry["graph"].replay()
```

**After** (수정):
```python
def _capture(self, input_ids, audio_mask, attention_mask=None, ...):
    static_input_ids = input_ids.clone()
    static_audio_mask = audio_mask.clone()
    static_attn_mask = attention_mask.clone() if attention_mask is not None else None
    # 모든 텐서를 static buffer로 클론
    ...
    with torch.cuda.graph(graph):
        static_output = self._original_forward(
            static_input_ids, static_audio_mask, attention_mask=static_attn_mask, ...
        )

def __call__(self, input_ids, audio_mask, attention_mask=None, ...):
    ...
    entry["static_input_ids"].copy_(input_ids)
    entry["static_audio_mask"].copy_(audio_mask)
    if attention_mask is not None and entry["static_attn_mask"] is not None:
        entry["static_attn_mask"].copy_(attention_mask)  # 모든 static buffer 업데이트
    entry["graph"].replay()
```

### 결과 비교

**수정 전** (Tier 3 Fast, 15문장):
| Runner | UTMOS | CER | Status |
|--------|-------|-----|--------|
| Faster | 3.30±1.28 | 0.40±0.65 | **FAIL** |
| Hybrid | 3.30±1.27 | 0.59±1.05 | **FAIL** |

**수정 후**:
| Runner | UTMOS | CER | Status |
|--------|-------|-----|--------|
| Faster | 3.84±0.53 | 0.09±0.13 | FAIL (delta 0.0507, 임계값 0.05 겨우 초과) |
| Hybrid | 3.76±0.61 | 0.11±0.14 | **PASS** |

---

## Bug 2: SageAttention — 잘못된 API 사용 및 설정

### 문제

SageAttention을 고수준 API(`sageattn()`)로 호출하면서 3가지 설정 오류가 있었습니다:

1. **`is_causal=False`** — OmniVoice의 Qwen3 backbone은 causal attention 사용. `False`로 설정하면 미래 토큰 참조
2. **attention_mask 직접 전달** — `sageattn()`은 arbitrary mask를 지원하지 않음. mask가 있으면 SDPA로 fallback 필요
3. **고수준 API 사용** — GPU 아키텍처별 최적 커널을 선택하지 못함

### 증상

```
Triton+Sage:  UTMOS 1.25±0.01, CER 1.01±0.18  ← 사용 불가 수준
Hybrid+Sage:  UTMOS 1.26±0.02, CER 1.09±0.33  ← 사용 불가 수준
```

UTMOS 1.25는 사실상 노이즈 수준이며, CER > 1.0은 생성된 텍스트가 원문보다 긴 가비지임을 의미합니다.

### 참조: ComfyUI-OmniVoice-TTS의 올바른 구현

[Saganaki22/ComfyUI-OmniVoice-TTS](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS)의 `nodes/sage_attention_patch.py`에서 정확한 구현을 발견:

```python
# GPU 아키텍처별 커널 선택
if arch_code >= 120:    # Blackwell → sageattn_qk_int8_pv_fp8_cuda
elif arch_code >= 90:   # Hopper   → sageattn_qk_int8_pv_fp8_cuda_sm90
elif arch_code == 89:   # Ada      → sageattn_qk_int8_pv_fp8_cuda
elif arch_code >= 80:   # Ampere   → sageattn_qk_int8_pv_fp16_cuda

# mask가 있으면 SDPA fallback
use_sage = (attention_mask is None and past_key_values is None)

# Causal attention + 저수준 CUDA 커널 직접 호출
attn_output = SAGE_ATTENTION_FUNCTION(
    q, k, v,
    tensor_layout="HND",
    is_causal=True,           # ← 핵심: causal=True
    qk_quant_gran="per_warp",
    pv_accum_dtype=pv_accum_dtype,  # GPU별 "fp32" 또는 "fp32+fp32"
)
```

### 수정 내용

**파일**: `src/omnivoice_triton/models/patching.py`

| 항목 | Before (버그) | After (수정) |
|------|-------------|------------|
| API | `sageattn()` (고수준) | `sageattn_qk_int8_pv_fp8_cuda` 등 (저수준) |
| `is_causal` | `False` | `True` |
| attention_mask | `sageattn(attn_mask=mask)` | mask 있으면 SDPA fallback |
| dtype 변환 | 없음 | `self.q_proj.weight.dtype`으로 변환 |
| `tensor_layout` | 미지정 | `"HND"` |
| `pv_accum_dtype` | 미지정 | GPU별 자동 선택 |
| GQA 처리 | 미지정 | SageAttention 내부 처리 (repeat_kv 불필요) |
| GPU 감지 | 없음 | SM80+ 아키텍처별 커널 자동 선택 |

### 결과 비교

**수정 전**:
| Runner | UTMOS | CER | Status |
|--------|-------|-----|--------|
| Triton+Sage | 1.25±0.01 | 1.01±0.18 | **FAIL** (가비지) |
| Hybrid+Sage | 1.26±0.02 | 1.09±0.33 | **FAIL** (가비지) |

**수정 후**:
| Runner | UTMOS | CER | Status |
|--------|-------|-----|--------|
| Triton+Sage | 3.81±0.58 | 0.11±0.16 | **PASS** |
| Hybrid+Sage | 3.87±0.54 | 0.10±0.13 | **PASS** |

---

## 최종 3-Tier 검증 결과 (수정 후)

**Hardware**: RTX 5090, CUDA 12.8, PyTorch 2.8, Python 3.12

### Tier 1: 커널 정확성 — 60/60 PASSED (8.0s)

3개 Triton 커널(RMSNorm, SwiGLU, FusedAddRMSNorm)이 모두 PyTorch 레퍼런스 대비 정확도, 결정성, 수치 안정성 테스트 통과.

### Tier 2: 모델 패리티 — 4/4 PASSED (34.4s)

| Runner | 오디오 길이 | max_abs | 추론 시간 | VRAM |
|--------|-----------|---------|-----------|------|
| Base | 2.99s | 0.5000 | 3.863s | 1.95GB |
| Triton | 2.75s | 0.5000 | 0.805s | 1.95GB |
| Faster | 3.11s | 0.5000 | 0.368s | 1.97GB |
| Hybrid | 3.12s | 0.5000 | 0.267s | 1.98GB |

### Tier 3: E2E 품질 — 5/6 PASSED (146s)

| Runner | UTMOS | CER | Speaker Sim | Status |
|--------|-------|-----|-------------|--------|
| Base (ref) | 3.77±0.63 | 0.14±0.18 | — | ref |
| Triton | 3.85±0.55 | 0.11±0.14 | 0.99 | **PASS** |
| Triton+Sage | 3.81±0.58 | 0.11±0.16 | 0.99 | **PASS** |
| Faster | 3.84±0.53 | 0.09±0.13 | 0.99 | FAIL* |
| Hybrid | 3.76±0.61 | 0.11±0.14 | 0.99 | **PASS** |
| Hybrid+Sage | 3.87±0.54 | 0.10±0.13 | 0.99 | **PASS** |

*Faster의 FAIL: CER delta 0.0507 (임계값 0.05를 0.0007 초과). qwen3-tts-triton의 Faster(0.0571)와 동일한 marginal fail 패턴. Full 모드(whisper-large-v3, 3회/문장, Mann-Whitney)에서 PASS 예상.

---

## 교훈

1. **CUDA Graph의 모든 GPU 텐서 인자는 static buffer로 관리해야 한다.** 함수 시그니처의 Optional 파라미터도 예외 없음. PyTorch CUDA allocator는 비-static 텐서의 메모리를 언제든 회수할 수 있으며, 이로 인한 corruption은 에너지(RMS) 기반 검사로 탐지 불가.

2. **SageAttention은 고수준 API가 아닌 저수준 CUDA 커널을 직접 사용해야 한다.** GPU 아키텍처별 최적 커널 선택, `is_causal` 설정, attention mask fallback 처리가 필수.

3. **TTS 품질 검증은 반드시 ASR 기반 CER/UTMOS를 포함해야 한다.** 오디오 에너지, 길이, 파형 통계만으로는 "올바른 언어의 올바른 내용"을 생성하는지 확인할 수 없음.
