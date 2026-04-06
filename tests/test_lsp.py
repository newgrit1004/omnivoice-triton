"""LSP (Liskov Substitution Principle) 테스트.

하위 타입은 상위 타입으로 치환 가능해야 한다.
- 모든 Animal 인스턴스는 동일한 인터페이스를 제공

uv run pytest tests/test_lsp.py -v
"""

import pytest

from tests.conftest import create_animal
from tests.params import LSP_ANIMAL_SUBSTITUTION_CASES, AnimalData
from zoo import Animal, Zoo


@pytest.mark.usefixtures("setup_class_fixture")
class TestLiskovSubstitutionPrinciple:
    """LSP 원칙 테스트 클래스."""

    @pytest.mark.parametrize("animal_data", LSP_ANIMAL_SUBSTITUTION_CASES)
    def test_all_animals_are_substitutable(self, animal_data: AnimalData) -> None:
        """모든 동물은 Animal 타입으로 치환 가능하다.

        Args:
            animal_data: 동물 데이터.
        """
        animal = create_animal(animal_data)

        # 모든 Animal은 동일한 인터페이스를 제공
        assert isinstance(animal, Animal)
        assert hasattr(animal, "speak")
        assert hasattr(animal, "is_adult")
        assert hasattr(animal, "name")
        assert hasattr(animal, "species")
        assert hasattr(animal, "age")

    def test_animal_interface_consistency(
        self, lion: Animal, baby_elephant: Animal, monkey: Animal
    ) -> None:
        """모든 동물은 일관된 인터페이스를 제공한다.

        Args:
            lion: 사자 fixture.
            baby_elephant: 아기 코끼리 fixture.
            monkey: 원숭이 fixture.
        """
        animals = [lion, baby_elephant, monkey]

        for animal in animals:
            # speak() 메서드는 문자열 반환
            sound = animal.speak()
            assert isinstance(sound, str)

            # is_adult() 메서드는 bool 반환
            is_adult = animal.is_adult()
            assert isinstance(is_adult, bool)

    def test_animals_can_be_used_interchangeably(self, seoul_zoo: Zoo) -> None:
        """동물들은 상호 교환 가능하다.

        Args:
            seoul_zoo: 서울동물원 fixture.
        """

        # 다양한 동물을 동일한 방식으로 처리
        animals_data = LSP_ANIMAL_SUBSTITUTION_CASES

        for animal_data in animals_data:
            animal = create_animal(animal_data)
            seoul_zoo.add_animal(animal)

        # 모든 동물이 Zoo에서 동일하게 처리됨
        assert seoul_zoo.count_animals() == len(animals_data)

    @pytest.mark.parametrize(
        "name,species,age",
        [
            ("Lion1", "lion", 5),
            ("Elephant1", "elephant", 15),
            ("Monkey1", "monkey", 6),
        ],
    )
    def test_animal_behavior_consistency(
        self, name: str, species: str, age: int
    ) -> None:
        """동물 행동은 일관성을 유지한다.

        Args:
            name: 동물 이름.
            species: 동물 종.
            age: 나이.
        """
        animal1 = Animal(name=name, species=species, age=age)
        animal2 = Animal(name=name, species=species, age=age)

        # 동일한 입력에 대해 동일한 결과
        assert animal1.speak() == animal2.speak()
        assert animal1.is_adult() == animal2.is_adult()
