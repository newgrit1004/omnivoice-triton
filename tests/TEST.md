# 테스트 구조 가이드

SOLID 원칙 기반 테스트 구조 설명

## 📁 디렉토리 구조

```
tests/
├── conftest.py          # pytest fixtures (class-scoped, function-scoped)
├── params.py            # 테스트 데이터 (dataclass, config, parametrize 데이터)
├── test_srp.py          # Single Responsibility Principle 테스트
├── test_ocp.py          # Open/Closed Principle 테스트
├── test_lsp.py          # Liskov Substitution Principle 테스트
├── test_isp.py          # Interface Segregation Principle 테스트
├── test_dip.py          # Dependency Inversion Principle 테스트
└── TEST.md              # 이 문서
```

---

## 🎯 SOLID 원칙별 테스트

### 1️⃣ SRP (Single Responsibility Principle) - test_srp.py

**원칙**: 각 클래스는 단 하나의 책임만 가져야 한다.

**테스트 대상**:
- `AnimalSoundService`: 동물 소리 관리만 담당
- `AgeValidator`: 나이 검증만 담당

**테스트 내용**:
```python
# 소리 서비스는 소리 관련 기능만 제공
@pytest.mark.parametrize("species,expected_sound", SRP_SOUND_TEST_CASES)
def test_animal_sound_service_responsibility(...)

# 나이 검증 서비스는 나이 관련 기능만 제공
@pytest.mark.parametrize("species,age,is_adult", SRP_AGE_TEST_CASES)
def test_age_validator_responsibility(...)
```

**주요 검증**:
- ✅ 각 서비스가 단일 책임만 수행
- ✅ 메서드 이름이 책임과 일치
- ✅ 다른 책임과 혼합되지 않음

---

### 2️⃣ OCP (Open/Closed Principle) - test_ocp.py

**원칙**: 확장에는 열려있고 수정에는 닫혀있어야 한다.

**테스트 대상**:
- `Animal`: 새로운 종 추가 시 클래스 수정 불필요
- `Zoo`: 새로운 동물원 타입 추가 시 클래스 수정 불필요

**테스트 내용**:
```python
# 새로운 동물 추가 시 Animal 클래스 수정 불필요
@pytest.mark.parametrize("animal_data", OCP_ANIMAL_TEST_DATA)
def test_animal_extension_without_modification(...)

# 새로운 종이 코드 변경 없이 추가 가능
def test_new_species_can_be_added_without_code_change(...)
```

**주요 검증**:
- ✅ 새로운 종 추가 시 기존 코드 수정 불필요
- ✅ 확장 가능한 구조 유지
- ✅ 다형성 지원

---

### 3️⃣ LSP (Liskov Substitution Principle) - test_lsp.py

**원칙**: 하위 타입은 상위 타입으로 치환 가능해야 한다.

**테스트 대상**:
- `Animal`: 모든 인스턴스가 동일한 인터페이스 제공

**테스트 내용**:
```python
# 모든 동물은 Animal 타입으로 치환 가능
@pytest.mark.parametrize("animal_data", LSP_ANIMAL_SUBSTITUTION_CASES)
def test_all_animals_are_substitutable(...)

# 모든 동물은 일관된 인터페이스 제공
def test_animal_interface_consistency(...)
```

**주요 검증**:
- ✅ 모든 Animal 인스턴스가 동일한 인터페이스 제공
- ✅ 상호 교환 가능
- ✅ 행동 일관성 유지

---

### 4️⃣ ISP (Interface Segregation Principle) - test_isp.py

**원칙**: 클라이언트는 사용하지 않는 인터페이스에 의존하지 않아야 한다.

**테스트 대상**:
- `Speakable`: 소리 내는 기능만 제공
- `Ageable`: 나이 관련 기능만 제공

**테스트 내용**:
```python
# Speakable 인터페이스는 소리 기능만 제공
@pytest.mark.parametrize("animal_data,expected_sound", ISP_SPEAKABLE_CASES)
def test_speakable_interface_isolation(...)

# Ageable 인터페이스는 나이 기능만 제공
@pytest.mark.parametrize("animal_data,expected_adult", ISP_AGEABLE_CASES)
def test_ageable_interface_isolation(...)
```

**주요 검증**:
- ✅ 인터페이스가 분리되어 있음
- ✅ 클라이언트가 필요한 인터페이스만 사용
- ✅ 불필요한 의존성 없음

---

### 5️⃣ DIP (Dependency Inversion Principle) - test_dip.py

**원칙**: 고수준 모듈은 저수준 모듈에 의존하지 않아야 한다. 둘 다 추상화에 의존해야 한다.

**테스트 대상**:
- `Zoo`: 구체 클래스가 아닌 `AnimalRepository` 인터페이스에 의존
- `AnimalRepository`: 독립적으로 동작 가능

**테스트 내용**:
```python
# Zoo는 구체 클래스가 아닌 추상화에 의존
@pytest.mark.parametrize("zoo_config", DIP_ZOO_CONFIGS)
def test_zoo_depends_on_abstraction(...)

# Repository는 Zoo에 주입 가능
def test_repository_can_be_injected(...)
```

**주요 검증**:
- ✅ 의존성 주입 가능
- ✅ 고수준 모듈이 저수준 모듈에 직접 의존하지 않음
- ✅ 추상화를 통한 느슨한 결합

---

## 🛠️ 테스트 구성 요소

### conftest.py

pytest 공통 fixtures 정의

**Class-scoped fixture**:
```python
@pytest.fixture(scope="class")
def setup_class_fixture(request: pytest.FixtureRequest) -> None:
    """클래스 레벨 setup/teardown."""
    print(f"\n[Setup] {request.cls.__name__} 테스트 클래스 시작")
    yield
    print(f"\n[Teardown] {request.cls.__name__} 테스트 클래스 종료")
```

**Function-scoped fixtures**:
- `lion()`: 성체 사자
- `baby_elephant()`: 아기 코끼리
- `monkey()`: 성체 원숭이
- `seoul_zoo()`: 빈 서울동물원
- `everland_zoo()`: 빈 에버랜드동물원
- `populated_seoul_zoo()`: 동물이 있는 서울동물원
- `empty_repository()`: 빈 저장소
- `populated_repository()`: 동물이 있는 저장소

### params.py

테스트 데이터 정의 (dataclass, config, parametrize 데이터)

**Dataclass 정의**:
```python
@dataclass(frozen=True)
class AnimalData:
    """동물 테스트 데이터."""
    name: str
    species: str
    age: int

@dataclass(frozen=True)
class ZooConfig:
    """동물원 설정."""
    name: str
    capacity: int = 100
```

**Parametrize 데이터**:
- `SRP_SOUND_TEST_CASES`: 소리 테스트 케이스
- `SRP_AGE_TEST_CASES`: 나이 검증 테스트 케이스
- `OCP_ANIMAL_TEST_DATA`: 동물 확장 테스트 데이터
- `LSP_ANIMAL_SUBSTITUTION_CASES`: 치환 가능성 테스트 데이터
- `ISP_SPEAKABLE_CASES`: Speakable 인터페이스 테스트 데이터
- `ISP_AGEABLE_CASES`: Ageable 인터페이스 테스트 데이터
- `DIP_ZOO_CONFIGS`: 동물원 설정 데이터
- `DIP_REPOSITORY_TEST_ANIMALS`: Repository 테스트 데이터

---

## 🚀 테스트 실행

### 전체 테스트 실행
```bash
make test
# 또는
uv run pytest
```

### 특정 SOLID 원칙 테스트
```bash
# SRP 테스트만 실행
uv run pytest tests/test_srp.py -v

# OCP 테스트만 실행
uv run pytest tests/test_ocp.py -v

# LSP 테스트만 실행
uv run pytest tests/test_lsp.py -v

# ISP 테스트만 실행
uv run pytest tests/test_isp.py -v

# DIP 테스트만 실행
uv run pytest tests/test_dip.py -v
```

### 커버리지 포함 실행
```bash
make test-cov
# 또는
uv run pytest --cov
```

### 커버리지 리포트 보기
```bash
make cov-report
# 브라우저로 htmlcov/index.html 자동 오픈
```

---

## 📊 테스트 통계

| 테스트 파일 | 원칙 | 테스트 수 | 주요 검증 사항 |
|------------|------|----------|---------------|
| test_srp.py | SRP | 5 | 단일 책임 원칙 |
| test_ocp.py | OCP | 4 | 확장성 |
| test_lsp.py | LSP | 4 | 치환 가능성 |
| test_isp.py | ISP | 4 | 인터페이스 분리 |
| test_dip.py | DIP | 5 | 의존성 역전 |
| **합계** | - | **22** | - |

---

## 💡 테스트 작성 가이드

### 1. Class-based 테스트
모든 SOLID 테스트는 클래스 기반으로 작성:
```python
@pytest.mark.usefixtures("setup_class_fixture")
class TestSingleResponsibility:
    """SRP 원칙 테스트 클래스."""

    def test_something(self) -> None:
        """테스트 설명."""
        assert True
```

### 2. Parametrize 사용
반복적인 테스트는 parametrize 활용:
```python
@pytest.mark.parametrize("species,expected_sound", SRP_SOUND_TEST_CASES)
def test_animal_sound(self, species: str, expected_sound: str) -> None:
    """여러 케이스를 한 번에 테스트."""
    sound = AnimalSoundService.get_sound(species)
    assert sound == expected_sound
```

### 3. Fixture 활용
conftest.py의 fixtures 적극 활용:
```python
def test_with_fixture(self, lion: Animal, seoul_zoo: Zoo) -> None:
    """Fixture로 테스트 데이터 재사용."""
    seoul_zoo.add_animal(lion)
    assert seoul_zoo.count_animals() == 1
```

---

## 🎓 학습 포인트

1. **SOLID 원칙 이해**: 각 원칙이 왜 중요한지 테스트를 통해 학습
2. **pytest 고급 기능**: parametrize, fixtures, class-based tests
3. **테스트 구조화**: params.py, conftest.py를 통한 체계적인 관리
4. **코드 품질**: 테스트를 통한 설계 검증

---

## 📚 참고 자료

- [SOLID 원칙](https://en.wikipedia.org/wiki/SOLID)
- [pytest 문서](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
