# 기여 가이드

프로젝트에 기여해주셔서 감사합니다! 아래 가이드라인을 따라주세요.

## 개발 환경 설정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd omnivoice-triton

# 2. 개발 환경 설정
make setup

# 3. pre-commit hooks 설치 (자동 실행됨)
# make setup이 자동으로 실행합니다
```

## 개발 워크플로우

### 1. 브랜치 전략

- `main`: 안정적인 릴리스 브랜치
- `feature/*`: 새 기능 개발
- `fix/*`: 버그 수정
- `docs/*`: 문서 업데이트

```bash
# 새 기능 브랜치 생성
git checkout -b feature/my-new-feature
```

### 2. 코드 작성

```bash
# 코드 포맷팅
make format

# 린트 체크
make lint

# 타입 체크
make typecheck

# 테스트 실행
make test
```

### 3. 커밋 메시지

커밋 메시지는 `.gitmessage.txt` 템플릿을 따릅니다:

```
[타입] 제목 (50자 이내)

본문 (72자 줄바꿈)
- 무엇을 변경했는지
- 왜 변경했는지

관련 이슈: #123
```

**타입**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 업데이트
- `style`: 코드 포맷팅
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 변경

### 4. Pull Request

1. 최신 main 브랜치와 동기화
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/my-new-feature
   git rebase main
   ```

2. 변경사항 푸시
   ```bash
   git push origin feature/my-new-feature
   ```

3. GitHub에서 Pull Request 생성

## 품질 기준

### 코드 품질

- ✅ Ruff 린트 통과 (`make lint`)
- ✅ Ruff 포맷 적용 (`make format`)
- ✅ Ty 타입 체크 통과 (`make typecheck`)
- ✅ 테스트 커버리지 ≥80% (`make test-cov`)

### 테스트 작성

- 모든 새 기능에 테스트 추가
- 버그 수정 시 재발 방지 테스트 작성
- `pytest` 사용
- 테스트 함수명: `test_<기능>_<시나리오>`

예시:
```python
import pytest

def test_get_name_basic() -> None:
    """기본 동작 테스트."""
    result = get_name("Alice")
    assert result == "내 이름은 Alice"

@pytest.mark.parametrize("name,expected", [
    ("Bob", "내 이름은 Bob"),
    ("Charlie", "내 이름은 Charlie"),
])
def test_get_name_parametrize(name: str, expected: str) -> None:
    """파라미터화 테스트."""
    assert get_name(name) == expected
```

### 문서화

- 모든 public 함수/클래스에 docstring 작성
- Google 스타일 docstring 사용
- README.md 업데이트 (필요 시)

예시:
```python
def get_name(name: str) -> str:
    """사용자 이름 메시지 생성.

    Args:
        name: 사용자 이름.

    Returns:
        포맷된 이름 메시지.

    Examples:
        >>> get_name("Alice")
        '내 이름은 Alice'
    """
    return f"내 이름은 {name}"
```

## 코드 리뷰

- 모든 PR은 최소 1명의 승인 필요
- 리뷰어는 24시간 내 응답 권장
- 건설적인 피드백 제공
- CI 통과 확인

## 문의

질문이나 제안사항은 이슈를 생성해주세요.
