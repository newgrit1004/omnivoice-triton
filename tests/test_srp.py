"""SRP (Single Responsibility Principle) 테스트.

각 클래스는 단 하나의 책임만 가져야 한다.
- AnimalSoundService: 동물 소리 관리만 담당
- AgeValidator: 나이 검증만 담당

uv run pytest tests/test_srp.py -v
"""

import pytest

from tests.params import SRP_AGE_TEST_CASES, SRP_SOUND_TEST_CASES
from zoo import AgeValidator, AnimalSoundService


@pytest.mark.usefixtures("setup_class_fixture")
class TestSingleResponsibility:
    """SRP 원칙 테스트 클래스."""

    @pytest.mark.parametrize("species,expected_sound", SRP_SOUND_TEST_CASES)
    def test_animal_sound_service_responsibility(
        self, species: str, expected_sound: str
    ) -> None:
        """AnimalSoundService는 소리 관리만 담당한다.

        Args:
            species: 동물 종.
            expected_sound: 예상 소리.
        """
        sound = AnimalSoundService.get_sound(species)
        assert sound == expected_sound

    def test_sound_service_has_single_responsibility(self) -> None:
        """AnimalSoundService는 오직 소리 관련 기능만 제공한다."""
        # 클래스 메서드 확인
        public_methods = [m for m in dir(AnimalSoundService) if not m.startswith("_")]
        sound_related = [m for m in public_methods if "sound" in m.lower()]

        # 소리 관련 메서드만 존재해야 함
        assert len(sound_related) > 0
        assert "get_sound" in public_methods

    @pytest.mark.parametrize("species,age,is_adult", SRP_AGE_TEST_CASES)
    def test_age_validator_responsibility(
        self, species: str, age: int, is_adult: bool
    ) -> None:
        """AgeValidator는 나이 검증만 담당한다.

        Args:
            species: 동물 종.
            age: 나이.
            is_adult: 성체 여부.
        """
        result = AgeValidator.is_adult(species, age)
        assert result == is_adult

    def test_age_validator_has_single_responsibility(self) -> None:
        """AgeValidator는 오직 나이 검증 기능만 제공한다."""
        # 클래스 메서드 확인
        public_methods = [m for m in dir(AgeValidator) if not m.startswith("_")]

        # 나이 관련 메서드만 존재해야 함
        assert "is_adult" in public_methods
        assert "get_adult_threshold" in public_methods

    @pytest.mark.parametrize(
        "species,expected_threshold",
        [
            ("lion", 3),
            ("elephant", 10),
            ("monkey", 4),
            ("bird", 1),
            ("unknown", 5),  # 기본값
        ],
    )
    def test_age_validator_threshold_logic(
        self, species: str, expected_threshold: int
    ) -> None:
        """AgeValidator는 올바른 성체 기준을 반환한다.

        Args:
            species: 동물 종.
            expected_threshold: 예상 성체 기준 나이.
        """
        threshold = AgeValidator.get_adult_threshold(species)
        assert threshold == expected_threshold
