"""ISP (Interface Segregation Principle) 테스트.

클라이언트는 사용하지 않는 인터페이스에 의존하지 않아야 한다.
- Speakable: 소리 내는 기능만 제공
- Ageable: 나이 관련 기능만 제공

uv run pytest tests/test_isp.py -v
"""

import pytest

from tests.params import ISP_AGEABLE_CASES, ISP_SPEAKABLE_CASES, AnimalData
from zoo import Animal


@pytest.mark.usefixtures("setup_class_fixture")
class TestInterfaceSegregationPrinciple:
    """ISP 원칙 테스트 클래스."""

    @pytest.mark.parametrize("animal_data,expected_sound", ISP_SPEAKABLE_CASES)
    def test_speakable_interface_isolation(
        self, animal_data: AnimalData, expected_sound: str
    ) -> None:
        """Speakable 인터페이스는 소리 기능만 제공한다.

        Args:
            animal_data: 동물 데이터.
            expected_sound: 예상 소리.
        """
        animal = Animal(
            name=animal_data.name,
            species=animal_data.species,
            age=animal_data.age,
        )

        # Speakable 인터페이스만 사용
        sound = animal.speak()
        assert sound == expected_sound
        assert isinstance(sound, str)

    @pytest.mark.parametrize("animal_data,expected_adult", ISP_AGEABLE_CASES)
    def test_ageable_interface_isolation(
        self, animal_data: AnimalData, expected_adult: bool
    ) -> None:
        """Ageable 인터페이스는 나이 기능만 제공한다.

        Args:
            animal_data: 동물 데이터.
            expected_adult: 예상 성체 여부.
        """
        animal = Animal(
            name=animal_data.name,
            species=animal_data.species,
            age=animal_data.age,
        )

        # Ageable 인터페이스만 사용
        is_adult = animal.is_adult()
        assert is_adult == expected_adult
        assert isinstance(is_adult, bool)

    def test_interface_independence(self, lion: Animal) -> None:
        """인터페이스는 서로 독립적이다.

        Args:
            lion: 사자 fixture.
        """
        # Speakable 인터페이스만 사용 가능
        sound = lion.speak()
        assert isinstance(sound, str)

        # Ageable 인터페이스만 사용 가능
        is_adult = lion.is_adult()
        assert isinstance(is_adult, bool)

    @pytest.mark.parametrize(
        "use_speak,use_age",
        [
            (True, False),  # Speakable만 사용
            (False, True),  # Ageable만 사용
            (True, True),  # 둘 다 사용
        ],
    )
    def test_clients_can_use_specific_interfaces(
        self, lion: Animal, use_speak: bool, use_age: bool
    ) -> None:
        """클라이언트는 필요한 인터페이스만 사용할 수 있다.

        Args:
            lion: 사자 fixture.
            use_speak: Speakable 사용 여부.
            use_age: Ageable 사용 여부.
        """
        if use_speak:
            sound = lion.speak()
            assert isinstance(sound, str)

        if use_age:
            is_adult = lion.is_adult()
            assert isinstance(is_adult, bool)
