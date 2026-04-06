"""DIP (Dependency Inversion Principle) 테스트.

고수준 모듈은 저수준 모듈에 의존하지 않아야 한다. 둘 다 추상화에 의존해야 한다.
- Zoo는 구체적인 저장소 구현이 아닌 AnimalRepository 인터페이스에 의존

uv run pytest tests/test_dip.py -v
"""

import pytest

from tests.params import DIP_REPOSITORY_TEST_ANIMALS, DIP_ZOO_CONFIGS, ZooConfig
from zoo import Animal, AnimalRepository, Zoo


@pytest.mark.usefixtures("setup_class_fixture")
class TestDependencyInversionPrinciple:
    """DIP 원칙 테스트 클래스."""

    @pytest.mark.parametrize("zoo_config", DIP_ZOO_CONFIGS)
    def test_zoo_depends_on_abstraction(self, zoo_config: ZooConfig) -> None:
        """Zoo는 구체 클래스가 아닌 추상화에 의존한다.

        Args:
            zoo_config: 동물원 설정.
        """
        # 기본 Repository 사용
        zoo1 = Zoo(name=zoo_config.name)
        assert zoo1.name == zoo_config.name

        # 커스텀 Repository 주입 가능 (의존성 역전)
        custom_repo = AnimalRepository()
        zoo2 = Zoo(name=zoo_config.name, repository=custom_repo)
        assert zoo2.name == zoo_config.name

    def test_repository_can_be_injected(self) -> None:
        """Repository는 Zoo에 주입 가능하다."""
        # Repository 생성 및 데이터 추가
        repo = AnimalRepository()
        for animal_data in DIP_REPOSITORY_TEST_ANIMALS:
            repo.add(
                Animal(
                    name=animal_data.name,
                    species=animal_data.species,
                    age=animal_data.age,
                )
            )

        # Zoo에 Repository 주입
        zoo = Zoo(name="Test Zoo", repository=repo)

        # 주입된 Repository 사용 확인
        assert zoo.count_animals() == len(DIP_REPOSITORY_TEST_ANIMALS)

    def test_repository_abstraction_independence(
        self, empty_repository: AnimalRepository
    ) -> None:
        """Repository는 독립적으로 동작한다.

        Args:
            empty_repository: 빈 저장소 fixture.
        """
        # Repository 직접 사용
        animal = Animal(name="Test", species="lion", age=3)
        empty_repository.add(animal)

        assert len(empty_repository.get_all()) == 1
        assert empty_repository.find_by_species("lion")[0].name == "Test"

    @pytest.mark.parametrize(
        "zoo_name",
        ["Seoul Zoo", "Everland Zoo"],
    )
    def test_multiple_zoos_share_repository(
        self, zoo_name: str, populated_repository: AnimalRepository
    ) -> None:
        """여러 동물원이 동일한 Repository를 공유할 수 있다.

        Args:
            zoo_name: 동물원 이름.
            populated_repository: 동물이 있는 저장소 fixture.
        """
        # 동일한 Repository를 공유하는 Zoo 생성
        zoo = Zoo(name=zoo_name, repository=populated_repository)

        # 공유된 데이터 접근 가능
        assert zoo.count_animals() == len(DIP_REPOSITORY_TEST_ANIMALS)

    def test_repository_encapsulation(
        self, populated_repository: AnimalRepository
    ) -> None:
        """Repository는 내부 구현을 캡슐화한다.

        Args:
            populated_repository: 동물이 있는 저장소 fixture.
        """
        # get_all()은 복사본을 반환해야 함
        animals1 = populated_repository.get_all()
        animals2 = populated_repository.get_all()

        # 복사본이므로 다른 객체
        assert animals1 is not animals2
        assert len(animals1) == len(animals2)
