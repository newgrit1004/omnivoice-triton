# 코드 품질 도구 설정 가이드

이 가이드는 코드 품질 분석을 위한 complexipy와 skylos의 설정 및 사용법을 다룹니다.

## 개요

| 도구 | 목적 | 언어 |
|------|------|------|
| **complexipy** | 인지 복잡도 분석 | Rust |
| **skylos** | 데드 코드 감지 + 보안 스캐닝 | Python |

## Complexipy (인지 복잡도)

### 인지 복잡도란?

인지 복잡도는 코드가 단순히 구조적으로 복잡한 것이 아니라, **이해하기** 얼마나 어려운지를 측정합니다. 다음을 고려합니다:
- 중첩 깊이
- 논리 연산자
- 제어 흐름 구조
- 재귀

### 설치

```bash
uv add --dev complexipy
```

### 기본 사용법

```bash
# 전체 프로젝트 분석
uv run complexipy .

# 특정 디렉토리 분석
uv run complexipy src/

# 임계값 설정
uv run complexipy . --max-complexity-allowed 15
```

### 임계값 가이드라인

| 임계값 | 수준 | 권장 대상 |
|--------|------|-----------|
| **10** | 엄격 | 새 프로젝트, 깨끗한 코드베이스 |
| **15** | 보통 (기본값) | 기존 코드베이스, 점진적 개선 |
| **20** | 느슨 | 레거시 코드, 초기 도입 |

### 점수 이해하기

| 점수 | 수준 | 조치 |
|------|------|------|
| 1-10 | 낮음 | 코드가 이해하기 쉬움 |
| 11-15 | 중간 | 단순화 고려 |
| 16-20 | 높음 | 리팩토링 권장 |
| 21+ | 심각 | 반드시 리팩토링 |

### 복잡도 요인

| 요인 | 추가되는 복잡도 |
|------|-----------------|
| `if`, `elif`, `else` | +1 (+ 중첩 보너스) |
| `for`, `while` | +1 (+ 중첩 보너스) |
| `try`, `except` | +1 |
| `and`, `or` | 각각 +1 |
| 재귀 | +1 |
| 중첩 보너스 | 레벨당 +1 |

### 출력 형식

```bash
# 콘솔 (기본)
uv run complexipy src/

# JSON
uv run complexipy src/ --output json

# CSV
uv run complexipy src/ --output csv
```

### 설정

#### 환경 변수
```bash
export COMPLEXIPY_MAX_COMPLEXITY=15
```

#### pyproject.toml
```toml
[tool.complexipy]
max-complexity-allowed = 15
```

### Pre-commit 훅

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/rohaquinlop/complexipy-pre-commit
    rev: v3.3.0
    hooks:
      - id: complexipy
        args: [--max-complexity-allowed, "15"]
```

---

## Skylos (데드 코드 감지)

### 데드 코드란?

데드 코드는 다음을 의미합니다:
- 미사용 함수
- 미사용 클래스
- 미사용 변수
- 미사용 임포트

### 설치

```bash
uv add --dev skylos
```

### 기본 사용법

```bash
# 데드 코드 찾기
uv run skylos .

# 신뢰도 임계값 설정
uv run skylos . --confidence 60

# 게이트 모드 (데드 코드 발견 시 exit 1)
uv run skylos . --gate
```

### 신뢰도 점수

| 점수 | 의미 | 권장 조치 |
|------|------|-----------|
| **90-100** | 확실히 미사용 | 즉시 삭제 안전 |
| **70-89** | 거의 확실히 미사용 | 간단히 검토 후 삭제 |
| **60-69** | 아마도 미사용 | 신중한 검토 필요 |
| **40-59** | 가능성 있는 미사용 | 거짓 양성 위험 높음 |
| **0-39** | 불확실 | 건너뛰기, 동적으로 사용될 가능성 |

### 보안 스캐닝

Skylos는 보안 이슈도 감지할 수 있습니다:

```bash
# 보안 스캐닝 활성화
uv run skylos . --danger --secrets

# 보안 플래그:
# --danger: 위험한 패턴 검사
# --secrets: 하드코딩된 비밀 검사
```

### 보안 스캔 감지 항목

| 유형 | 예시 |
|------|------|
| API 키 | `API_KEY = "sk-..."` |
| 비밀번호 | `password = "..."` |
| 토큰 | `token = "eyJ..."` |
| 개인 키 | `-----BEGIN PRIVATE KEY-----` |

### 프레임워크 인식 분석

Skylos는 일반적인 프레임워크 패턴을 자동으로 제외합니다:

| 프레임워크 | 자동 제외 |
|-----------|----------|
| **Django** | 모델, 뷰, 마이그레이션, 관리 명령 |
| **Flask** | 라우트 핸들러, 블루프린트 |
| **FastAPI** | 엔드포인트 함수, 의존성 |
| **Pytest** | 테스트 함수, 픽스처, conftest |

### 출력 형식

```bash
# 콘솔 (기본)
uv run skylos src/

# JSON (CI/CD용)
uv run skylos src/ --output json

# SARIF (GitHub 통합)
uv run skylos src/ --output sarif

# CSV (스프레드시트 분석)
uv run skylos src/ --output csv
```

### 설정

#### 환경 변수
```bash
export SKYLOS_CONFIDENCE=60
export SKYLOS_SECURITY=true  # --danger --secrets 활성화
```

### Pre-commit 훅 (커스텀)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: skylos-dead-code
        name: Skylos 데드 코드 검사
        entry: uv run skylos
        args: ['.', '-c', '80', '--gate']
        language: system
        pass_filenames: false
        types: [python]
```

---

## 통합 사용

### 권장 워크플로우

```
1. 먼저 데드 코드 감지 실행
   → 복잡도 분석 전 코드베이스 크기 축소

2. 복잡도 분석 실행
   → 남은 복잡한 함수에 집중

3. 순서대로 이슈 수정:
   a. 보안 이슈 (즉시)
   b. 데드 코드 (빠른 성과)
   c. 복잡한 함수 (리팩토링)

4. 테스트로 검증
   → 회귀 없음 확인
```

### Makefile 타겟

```makefile
# 개별 도구
make complexity    # complexipy 실행
make dead-code     # skylos 실행

# 통합 분석
make code-quality  # 두 도구 모두 실행
```

### CI/CD 통합

```yaml
# .github/workflows/quality.yml
name: 코드 품질

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: UV 설치
        uses: astral-sh/setup-uv@v4

      - name: 의존성 설치
        run: uv sync --all-extras --dev

      - name: 복잡도 검사
        run: uv run complexipy src/ --max-complexity-allowed 15

      - name: 데드 코드 검사
        run: uv run skylos src/ --gate --confidence 80
```

---

## 문제 해결

### Complexipy 이슈

**문제**: 단순 함수에서 거짓 양성
**해결**: 함수에 많은 조기 반환이 있는지 확인; 이것은 실제로 좋은 관행

**문제**: 모듈 수준 복잡도 보고됨
**해결**: Complexipy는 함수만 분석; 모듈 수준 코드는 점수화되지 않음

### Skylos 이슈

**문제**: 프레임워크 코드가 데드로 보고됨
**해결**: Skylos가 대부분의 프레임워크를 자동 감지; 프레임워크가 지원되는지 확인

**문제**: 동적 코드가 데드로 보고됨
**해결**: `getattr()`, `importlib` 등을 사용하는 코드는 사용 중으로 감지되지 않음; 낮은 신뢰도 임계값 사용

**문제**: 테스트 픽스처가 데드로 보고됨
**해결**: `conftest.py` 파일이 스캔 경로에 있는지 확인

---

## 빠른 참조

### Complexipy 명령어
```bash
uv run complexipy .                              # 기본 스캔
uv run complexipy src/ --max-complexity-allowed 10  # 엄격
uv run complexipy . --output json                # JSON 출력
```

### Skylos 명령어
```bash
uv run skylos .                    # 기본 스캔
uv run skylos . -c 80 --gate       # 높은 신뢰도 + 게이트
uv run skylos . --danger --secrets # 보안 스캔
uv run skylos . --output json      # JSON 출력
```

### 두 도구 모두
```bash
make code-quality                  # 둘 다 실행
make pre-commit                    # 전체 pre-commit 검사
```
