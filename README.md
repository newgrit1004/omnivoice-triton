# 현대적인 파이썬 프로젝트 템플릿

최신 Python 개발 도구를 사용한 프로젝트 템플릿입니다.

#   🚀 업그레이드 요약

| 항목         | 이전                     | 현재               | 성능 향상      |
|------------|------------------------|------------------|------------|
| 패키지 관리     | pip + requirements.txt | UV               | 10-100배 빠름 |
| 포맷팅        | Black                  | Ruff format      | 10-100배 빠름 |
| 린팅         | flake8 + pylint        | Ruff check       | 10-100배 빠름 |
| Import 정렬  | isort                  | Ruff (내장)        | -          |
| 타입 체킹      | mypy                   | Ty               | 10-20배 빠름  |
| Pre-commit | v4.4.0                 | v5.0.0           | -          |
| CI/CD      | ❌                      | ✅ GitHub Actions | -          |

## 주요 도구

- **패키지 관리**: UV (Astral)
  - 기존 pip/poetry보다 10-100배 빠른 패키지 관리
  - 통합된 프로젝트 관리 및 가상환경 관리

- **포맷팅 & 린팅**: Ruff (Astral)
  - Black + isort + flake8을 하나로 통합
  - Rust로 작성되어 매우 빠른 속도
  - 자동 수정 기능 (`--fix`)

- **타입 체킹**: Ty (Astral)
  - Rust 기반의 초고속 타입 체커
  - mypy/pyright 대비 10-20배 빠른 속도
  - gradual guarantee 지원 (기존 코드에 안전)

- **테스트**: pytest + pytest-cov
  - 코드 커버리지 리포트 자동 생성

- **CI/CD**: pre-commit + GitHub Actions
  - 커밋 전 자동 품질 체크
  - GitHub에서 자동화된 CI/CD 파이프라인

## 프로젝트 환경 세팅

### UV 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip (모든 플랫폼)
pip install uv
```

### 프로젝트 초기 설정

```bash
# UV를 사용한 프로젝트 설정 (가상환경 자동 생성)
make setup

# 또는 수동으로
uv sync --all-extras --dev
uv run pre-commit install
git config commit.template .gitmessage.txt
```

**수행 항목**:
- UV를 사용한 의존성 설치
- Git commit message 템플릿 적용
- pre-commit hooks 설치

## 개발 명령어

### 코드 품질 관리

```bash
# 코드 포맷팅 (Ruff)
make format              # 전체 프로젝트
make format DIR=src      # 특정 디렉토리

# 린팅 체크
make lint                # 전체 프로젝트
make lint-fix            # 자동 수정 포함

# 타입 체킹 (Ty)
make typecheck           # 전체 프로젝트
make typecheck DIR=src   # 특정 디렉토리

# 테스트 실행
make test                # 기본 테스트
make test-cov            # 커버리지 포함

# pre-commit 전체 실행
make pre-commit
```

### 프로젝트 관리

```bash
# 의존성 설치
make install

# 의존성 업데이트
make update

# 캐시 정리
make clean
```

## Git 워크플로우

### Commit 작성

```bash
git add .
git commit  # 템플릿 에디터가 열립니다
```

**pre-commit 자동 체크 순서**:
1. ✓ 대용량 파일 체크
2. ✓ JSON/YAML/TOML 문법 검증
3. ✓ 머지 충돌 체크
4. ✓ 파일 끝 줄바꿈 수정
5. ✓ 공백 문자 정리
6. ✓ **Ruff 린팅** (자동 수정)
7. ✓ **Ruff 포맷팅**
8. ✓ **Ty 타입 체킹**

### Commit 메시지 형식

```
[type]: [Subject]

[body]

[footer]
```

**타입 종류**:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅 (기능 변경 없음)
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/패키지 설정 등

**작성 규칙**:
- Subject: 50자 이하, 대문자 시작, 마침표 없음, 명령형
- Body: 선택사항, 72자로 줄바꿈
- Footer: 이슈 참조 (예: `Resolves: #123`)

## 도구 설정

### Ruff (포맷팅 & 린팅)

**설정**: `pyproject.toml`의 `[tool.ruff]`

- Line length: 88자
- Target: Python 3.12+
- 포맷팅: 자동 (`ruff format`)
- 린팅 룰:
  - E, W: pycodestyle (PEP 8)
  - F: pyflakes (오류 탐지)
  - I: isort (import 정렬)

**인라인 무시**:
```python
from .base import *  # noqa: F403
```

### Ty (타입 체킹)

**설정**: `pyproject.toml`의 `[tool.ty]`

- Python 버전: 3.12
- Gradual typing 지원
- Rust 기반 초고속 체킹

**인라인 무시**:
```python
def foo() -> str:
    return 123  # type: ignore
```

### pytest

**설정**: `pyproject.toml`의 `[tool.pytest.ini_options]`

- Python path: `src`
- 테스트 경로: `tests`
- 자동 커버리지 리포트 생성 (HTML, 터미널)

## CI/CD

### GitHub Actions

`.github/workflows/ci.yml`에서 자동화:

- Python 3.12, 3.13 매트릭스 테스트
- Ruff 린팅 & 포맷팅 체크
- Ty 타입 체킹
- pytest 테스트 + 커버리지
- Codecov 업로드 (선택사항)

### 로컬 pre-commit

모든 커밋 전 자동 실행:
- 파일 검증
- Ruff 자동 수정
- Ty 타입 체킹


## 프로젝트 구조

```
python-project-template/
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
├── src/
│   ├── __init__.py
│   └── main.py              # 메인 코드
├── tests/
│   └── test_main.py         # 테스트 코드
├── .gitignore
├── .gitmessage.txt          # Git 커밋 템플릿
├── .pre-commit-config.yaml  # pre-commit 설정
├── pyproject.toml           # 프로젝트 설정 (UV 기반)
├── uv.lock                  # UV 잠금 파일
├── Makefile                 # 개발 명령어 단축키
├── README.md
└── CLAUDE.md                # Claude Code 가이드
```

## 참고 자료

- **UV**: https://docs.astral.sh/uv/
- **Ruff**: https://docs.astral.sh/ruff/
- **Ty**: https://docs.astral.sh/ty/
- **pytest**: https://docs.pytest.org/
- **pre-commit**: https://pre-commit.com/
