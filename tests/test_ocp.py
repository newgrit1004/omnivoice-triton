"""OCP (Open/Closed Principle) 테스트.

확장에는 열려있고 수정에는 닫혀있어야 한다.
- Animal 클래스는 새로운 종 추가 시 클래스 수정 없이 확장 가능

uv run pytest tests/test_ocp.py -v
"""

import pytest

from tests.conftest import create_animal
from tests.params import OCP_ANIMAL_TEST_DATA, AnimalData
from zoo import Animal


@pytest.mark.usefixtures("setup_class_fixture")
class TestOpenClosedPrinciple:
    """OCP 원칙 테스트 클래스."""

    @pytest.mark.parametrize("animal_data", OCP_ANIMAL_TEST_DATA)
    def test_animal_extension_without_modification(
        self, animal_data: AnimalData
    ) -> None:
        """새로운 동물 추가 시 Animal 클래스 수정이 불필요하다.

        Args:
            animal_data: 동물 데이터.
        """
        # 새로운 동물 생성 (클래스 수정 없이 확장)
        animal = create_animal(animal_data)

        # 기본 동작 검증
        assert animal.name == animal_data.name
        assert animal.species == animal_data.species
        assert animal.age == animal_data.age

    def test_new_species_can_be_added_without_code_change(self) -> None:
        """새로운 종이 코드 변경 없이 추가 가능하다."""
        # 새로운 종 추가 (Animal 클래스 수정 없음)
        new_species = Animal(name="Nemo", species="fish", age=1)

        # 기본 동작 검증
        assert new_species.speak() == "..."  # 기본 소리
        assert isinstance(new_species.is_adult(), bool)  # 성체 판별 가능

    @pytest.mark.parametrize(
        "zoo_name",
        ["Seoul Zoo", "Everland Zoo"],
    )
    def test_multiple_zoo_types_supported(self, zoo_name: str, lion: Animal) -> None:
        """다양한 동물원 타입이 수정 없이 지원된다.

        Args:
            zoo_name: 동물원 이름.
            lion: 사자 fixture.
        """
        from zoo import Zoo

        # 다양한 동물원 생성 (Zoo 클래스 수정 없음)
        zoo = Zoo(name=zoo_name)
        zoo.add_animal(lion)

        assert zoo.name == zoo_name
        assert zoo.count_animals() == 1

    def test_animal_behavior_is_extensible(self, lion: Animal) -> None:
        """Animal 동작은 확장 가능하다.

        Args:
            lion: 사자 fixture.
        """
        # 기본 동작 확인
        assert hasattr(lion, "speak")
        assert hasattr(lion, "is_adult")

        # 새로운 속성 추가 가능 (확장 가능)
        lion.habitat = "Savanna"  # type: ignore
        assert lion.habitat == "Savanna"  # type: ignore
