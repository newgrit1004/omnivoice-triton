"""몽키패칭(Monkey Patching) 기법 테스트 - 초보자용.

몽키패칭이란?
런타임에 클래스나 모듈의 동작을 동적으로 변경하는 기법입니다.
주로 테스트에서 외부 의존성을 모킹(mocking)할 때 사용합니다.

주의사항:
- 프로덕션 코드에서는 사용을 자제하세요
- 테스트 간 격리를 위해 반드시 원복하세요
- pytest의 monkeypatch fixture 사용을 권장합니다

uv run pytest tests/test_monkey_patching.py -v
"""

import logging
from collections.abc import Generator

import pytest

from zoo import Animal, AnimalSoundService

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_state() -> Generator[None, None, None]:
    """각 테스트 후 상태를 원복하는 fixture.

    몽키패칭으로 인한 부작용을 방지합니다.
    """
    # 원본 상태 저장
    original_sounds = AnimalSoundService._sounds.copy()
    original_speak = Animal.speak

    yield  # 테스트 실행

    # 상태 원복
    AnimalSoundService._sounds = original_sounds
    Animal.speak = original_speak  # type: ignore[method-assign]


class TestMonkeyPatchingBasics:
    """몽키패칭 기초 예제."""

    def test_01_class_variable_patching(self) -> None:
        """예제 1: 클래스 변수를 수정하는 가장 간단한 방법.

        새로운 데이터를 추가하거나 기존 데이터를 변경합니다.
        """
        logger.info("클래스 변수 패칭 테스트")

        # 원본: lion은 "으르렁!" 소리
        lion = Animal(name="Simba", species="lion", age=5)
        assert lion.speak() == "으르렁!"

        # 패칭: 새로운 동물 추가
        AnimalSoundService._sounds["dragon"] = "크와앙!"

        dragon = Animal(name="Smaug", species="dragon", age=100)
        assert dragon.speak() == "크와앙!"

        logger.info("✅ 딕셔너리에 새 항목을 추가하여 동작 변경")

    def test_02_class_method_patching(self) -> None:
        """예제 2: 클래스 메서드를 완전히 교체하는 방법.

        모든 인스턴스의 동작이 변경됩니다.
        """
        logger.info("클래스 메서드 패칭 테스트")

        # 원본 메서드 저장
        original_speak = Animal.speak

        # 새로운 메서드 정의
        def loud_speak(self: Animal) -> str:
            """기존 소리에 이모지를 추가합니다."""
            sound = AnimalSoundService.get_sound(self.species)
            return f"🔊 {sound} 🔊"

        # 메서드 교체
        Animal.speak = loud_speak  # type: ignore[method-assign]

        # 모든 인스턴스가 영향받음
        lion = Animal(name="Simba", species="lion", age=5)
        elephant = Animal(name="Dumbo", species="elephant", age=3)

        assert lion.speak() == "🔊 으르렁! 🔊"
        assert elephant.speak() == "🔊 뿌우~ 🔊"

        # 원복
        Animal.speak = original_speak  # type: ignore[method-assign]

        logger.info("✅ 클래스 메서드를 교체하여 모든 인스턴스에 영향")

    def test_03_pytest_monkeypatch_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """예제 3: pytest의 monkeypatch fixture 사용 (권장).

        pytest가 제공하는 안전한 방법으로, 테스트 종료 시 자동으로 원복됩니다.
        """
        logger.info("pytest monkeypatch 테스트")

        # monkeypatch로 메서드 교체
        def mock_get_sound(species: str) -> str:
            """모든 동물이 같은 소리를 내도록 합니다."""
            return "테스트용 소리"

        monkeypatch.setattr(AnimalSoundService, "get_sound", mock_get_sound)

        # 모든 동물이 같은 소리
        lion = Animal(name="Simba", species="lion", age=5)
        elephant = Animal(name="Dumbo", species="elephant", age=3)

        assert lion.speak() == "테스트용 소리"
        assert elephant.speak() == "테스트용 소리"

        # 테스트 종료 시 자동으로 원복됨 (수동 원복 불필요)
        logger.info("✅ monkeypatch는 자동으로 원복되므로 가장 안전")

    def test_04_monkeypatch_add_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """예제 4: monkeypatch로 딕셔너리에 항목 추가.

        실용적인 예제: 테스트를 위해 임시 데이터를 추가합니다.
        """
        logger.info("monkeypatch 딕셔너리 추가 테스트")

        # 딕셔너리에 새 항목 추가
        monkeypatch.setitem(AnimalSoundService._sounds, "unicorn", "반짝반짝✨")
        monkeypatch.setitem(AnimalSoundService._sounds, "phoenix", "불타올라🔥")

        unicorn = Animal(name="Sparkle", species="unicorn", age=1)
        phoenix = Animal(name="Fawkes", species="phoenix", age=50)

        assert unicorn.speak() == "반짝반짝✨"
        assert phoenix.speak() == "불타올라🔥"

        logger.info("✅ 테스트를 위한 임시 데이터 추가")


class TestMonkeyPatchingPractical:
    """실전 활용 예제."""

    def test_mocking_for_testing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """실전 예제: 외부 의존성 모킹.

        실제 테스트에서 데이터베이스나 API 호출을 모킹할 때 사용합니다.
        """
        logger.info("외부 의존성 모킹 테스트")

        # 호출 횟수 추적
        call_count = 0

        def mock_get_sound(species: str) -> str:
            """모의(mock) 함수: 호출 횟수를 세고 항상 같은 값을 반환."""
            nonlocal call_count
            call_count += 1
            return "모킹된 소리"

        monkeypatch.setattr(AnimalSoundService, "get_sound", mock_get_sound)

        lion = Animal(name="Simba", species="lion", age=5)
        lion.speak()
        lion.speak()

        assert call_count == 2  # 2번 호출됨
        logger.info("✅ 모킹을 통해 외부 의존성 제어 및 검증")

    def test_error_simulation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """실전 예제: 에러 상황 시뮬레이션.

        에러 처리 로직을 테스트할 때 유용합니다.
        """
        logger.info("에러 시뮬레이션 테스트")

        def raise_error(*args: object, **kwargs: object) -> str:
            """항상 에러를 발생시킵니다."""
            raise RuntimeError("테스트용 에러")

        monkeypatch.setattr(AnimalSoundService, "get_sound", raise_error)

        lion = Animal(name="Simba", species="lion", age=5)

        # 에러가 발생하는지 확인
        with pytest.raises(RuntimeError, match="테스트용 에러"):
            lion.speak()

        logger.info("✅ 에러 상황을 인위적으로 만들어 테스트")


# ============================================================================
# 핵심 요약
# ============================================================================
"""
몽키패칭 활용 가이드:

1. 클래스 변수 수정 (test_01):
   - 가장 간단한 방법
   - 딕셔너리나 리스트에 데이터 추가/변경

2. 클래스 메서드 교체 (test_02):
   - 동작 자체를 변경
   - 모든 인스턴스에 영향
   - 반드시 원복 필요

3. pytest monkeypatch 사용 (test_03, test_04):
   - 가장 안전하고 권장되는 방법
   - 자동으로 원복됨
   - setattr: 속성/메서드 교체
   - setitem: 딕셔너리 항목 추가
   - delitem: 딕셔너리 항목 제거

4. 실전 활용 (test_mocking, test_error):
   - 외부 의존성 모킹
   - 에러 상황 테스트
   - 호출 횟수 검증

⚠️ 주의사항:
- 프로덕션 코드에서는 사용 자제
- 테스트에서만 사용
- 항상 원복 처리
- monkeypatch fixture 권장
"""
