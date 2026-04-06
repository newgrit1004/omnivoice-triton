# UV 패키지 매니저 완벽 가이드

## UV란?

Astral에서 만든 **초고속 Python 패키지 매니저**로, pip와 poetry를 대체합니다.
- Rust로 작성되어 10-100배 빠른 속도
- 의존성 해결, 가상환경 관리, 프로젝트 관리를 통합

---

## 가상환경 (Virtual Environment)

### UV가 생성하는 가상환경 이름

UV는 **`.venv`** (점으로 시작)를 기본 가상환경 이름으로 사용합니다.

```bash
# UV가 자동으로 .venv 생성
uv sync

# 확인
ls -la
# drwxr-xr-x  .venv/
```
### 가상환경 수동 활성화가 필요 없는 이유

**전통적인 방식 (pip):**
```bash
python -m venv venv
source venv/bin/activate  # 매번 필수!
pip install package
python main.py
```

**UV 방식:**
```bash
uv sync
uv run python main.py  # 자동으로 .venv 사용
```

UV는 `uv run` 명령어가 **자동으로 `.venv`를 찾아서 활성화**합니다.

---

## 패키지 설치 및 관리

### 새로운 패키지 설치 방법

**❌ 잘못된 방법:**
```bash
pip install requests  # pyproject.toml에 기록 안 됨!
uv sync              # requests가 삭제됨
```

**✅ 올바른 방법:**

#### 1. `uv add` 사용 (권장)
```bash
# 프로덕션 의존성
uv add requests

# 개발 의존성
uv add --dev pytest

# 자동으로 pyproject.toml 업데이트 + 설치
```

#### 2. `pyproject.toml` 직접 수정
```toml
[project]
dependencies = [
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]
```

```bash
# 변경사항 적용
uv sync
```

### 패키지 제거

```bash
# uv remove 사용
uv remove requests

# 또는 pyproject.toml에서 삭제 후
uv sync
```

---

## ModuleNotFoundError 발생 원리

### 왜 `python main.py`는 안 되고 `uv run python main.py`는 되는가?

**시나리오:**
```bash
uv add requests
python main.py  # ❌ ModuleNotFoundError: No module named 'requests'
uv run python main.py  # ✅ 정상 작동
```

**원인:**

1. **시스템 Python vs UV 가상환경**
   ```bash
   which python
   # /usr/bin/python  (시스템 Python, requests 없음)

   uv run which python
   # /path/to/project/.venv/bin/python  (가상환경, requests 있음)
   ```

2. **UV가 패키지를 설치하는 위치**
   ```
   프로젝트/
   ├── .venv/
   │   └── lib/python3.12/site-packages/
   │       └── requests/  ← 여기에 설치됨
   └── main.py
   ```

3. **해결 방법**

   **방법 1: uv run 사용 (권장)**
   ```bash
   uv run python main.py
   uv run pytest
   ```

   **방법 2: 가상환경 수동 활성화**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # 또는
   .venv\Scripts\activate.bat  # Windows

   python main.py  # 이제 작동
   ```

   **방법 3: Makefile 사용**
   ```bash
   make test  # 내부적으로 uv run pytest
   ```

---

## UV 주요 명령어

### 프로젝트 설정

```bash
# 새 프로젝트 생성
uv init my-project
cd my-project

# 기존 프로젝트 설정
uv sync  # pyproject.toml 읽어서 설치
```

### 의존성 관리

```bash
# 패키지 추가
uv add requests
uv add --dev pytest

# 패키지 제거
uv remove requests

# 의존성 업데이트
uv lock --upgrade  # uv.lock 갱신
uv sync            # 갱신된 버전 설치

# 현재 설치된 패키지 확인
uv pip list
```

### 실행 환경

```bash
# 가상환경에서 명령어 실행
uv run python main.py
uv run pytest
uv run black .

# Python REPL
uv run python

# 스크립트 직접 실행
uv run --script script.py
```

---

## pyproject.toml vs uv.lock

### pyproject.toml
**역할**: 프로젝트 메타데이터 + 의존성 요구사항

```toml
[project]
dependencies = [
    "requests>=2.31.0",  # 2.31.0 이상 아무거나
]
```

- 버전 범위 지정
- 사람이 직접 편집
- Git에 커밋

### uv.lock
**역할**: 정확한 버전 고정 (lockfile)

```
requests==2.31.0
certifi==2024.2.2
charset-normalizer==3.3.2
...
```

- 정확한 버전 고정
- UV가 자동 생성
- Git에 커밋 (재현성 보장)

**비유:**
- `pyproject.toml`: 레시피 (재료 대략적 양)
- `uv.lock`: 정확한 계량 (정확한 그램 수)

---

## 일반적인 워크플로우

### 1. 새 프로젝트 시작
```bash
uv init my-project
cd my-project
uv add requests fastapi
uv add --dev pytest black ruff
```

### 2. 기존 프로젝트 클론
```bash
git clone <repo>
cd <repo>
uv sync  # uv.lock 기반으로 정확히 동일한 버전 설치
```

### 3. 개발 중
```bash
# 새 패키지 필요
uv add pandas

# 코드 작성
vim src/main.py

# 테스트
uv run pytest

# 실행
uv run python src/main.py
```

### 4. 의존성 업데이트
```bash
uv lock --upgrade  # 최신 버전으로 lock 파일 갱신
uv sync            # 갱신된 버전 설치
```

---

## UV vs pip 명령어 비교표

| 작업 | pip | UV |
|------|-----|-----|
| 가상환경 생성 | `python -m venv venv` | 자동 (`.venv`) |
| 가상환경 활성화 | `source venv/bin/activate` | 불필요 (`uv run` 사용) |
| 패키지 설치 | `pip install requests` | `uv add requests` |
| 개발 패키지 설치 | `pip install -e .` | `uv sync --dev` |
| 패키지 제거 | `pip uninstall requests` | `uv remove requests` |
| 의존성 파일 | `requirements.txt` | `pyproject.toml` + `uv.lock` |
| 의존성 고정 | `pip freeze > requirements.txt` | `uv lock` |
| 전체 설치 | `pip install -r requirements.txt` | `uv sync` |
| 업데이트 | `pip install --upgrade requests` | `uv lock --upgrade && uv sync` |
| 실행 | `python main.py` (활성화 후) | `uv run python main.py` |

---

## 트러블슈팅

### Q1. `python main.py`로 실행했더니 ModuleNotFoundError
**A:** `uv run python main.py` 사용하거나 `.venv` 활성화

### Q2. 패키지 설치했는데 `uv sync` 하니까 사라짐
**A:** `uv add <package>` 또는 `pyproject.toml`에 추가 후 `uv sync`

### Q3. `.venv` 폴더가 너무 큼
**A:** 정상. 모든 의존성이 여기 설치됨. `.gitignore`에 포함되어 Git에는 안 올라감

### Q4. 다른 팀원과 다른 버전 설치됨
**A:** `uv.lock` 파일을 Git에 커밋. `uv sync`는 lock 파일 기반으로 동일 버전 설치

### Q5. UV 없이 프로젝트 실행 가능한가?
**A:** 가능. `pyproject.toml`은 PEP 621 표준이므로 pip도 인식:
```bash
pip install -e .
python main.py
```

---

## 고급 기능

### 특정 Python 버전 사용
```bash
uv venv --python 3.11
uv sync
```

### 의존성 그룹별 설치
```bash
uv sync --group docs  # docs 그룹만 설치
uv sync --all-groups  # 모든 그룹 설치
```

### 캐시 관리
```bash
uv cache clean  # 캐시 정리
uv cache dir    # 캐시 위치 확인
```

---

## 요약

✅ **UV 사용 시 기억할 핵심**

1. **가상환경**: `.venv` (자동 생성)
2. **활성화 불필요**: `uv run` 사용
3. **패키지 설치**: `uv add <package>`
4. **실행**: `uv run python main.py`
5. **동기화**: `uv sync` (팀원과 동일 환경)

❌ **하지 말아야 할 것**

1. ~~`pip install`~~ → `uv add` 사용
2. ~~`python main.py`~~ → `uv run python main.py` 사용
3. ~~`requirements.txt` 수정~~ → `pyproject.toml` 수정

---

## 참고 자료

- 공식 문서: https://docs.astral.sh/uv/
- GitHub: https://github.com/astral-sh/uv
