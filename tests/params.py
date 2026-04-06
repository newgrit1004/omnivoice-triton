"""테스트 파라미터 및 설정."""

from dataclasses import dataclass


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


# ============================================================================
# Single Responsibility Principle (SRP) 테스트 데이터
# ============================================================================

SRP_SOUND_TEST_CASES = [
    ("lion", "으르렁!"),
    ("elephant", "뿌우~"),
    ("monkey", "우끼끼!"),
    ("bird", "짹짹!"),
    ("unknown", "..."),
]

SRP_AGE_TEST_CASES = [
    ("lion", 3, True),  # 사자 성체 기준: 3세
    ("lion", 2, False),
    ("elephant", 10, True),  # 코끼리 성체 기준: 10세
    ("elephant", 9, False),
    ("monkey", 4, True),  # 원숭이 성체 기준: 4세
    ("bird", 1, True),  # 새 성체 기준: 1세
]


# ============================================================================
# Open/Closed Principle (OCP) 테스트 데이터
# ============================================================================

OCP_ANIMAL_TEST_DATA = [
    AnimalData(name="Simba", species="lion", age=5),
    AnimalData(name="Nala", species="lion", age=4),
    AnimalData(name="Dumbo", species="elephant", age=12),
    AnimalData(name="Coco", species="monkey", age=6),
]


# ============================================================================
# Liskov Substitution Principle (LSP) 테스트 데이터
# ============================================================================

LSP_ANIMAL_SUBSTITUTION_CASES = [
    AnimalData(name="Leo", species="lion", age=3),
    AnimalData(name="Ella", species="elephant", age=10),
    AnimalData(name="Max", species="monkey", age=4),
]


# ============================================================================
# Interface Segregation Principle (ISP) 테스트 데이터
# ============================================================================

ISP_SPEAKABLE_CASES = [
    (AnimalData(name="Leo", species="lion", age=3), "으르렁!"),
    (AnimalData(name="Ella", species="elephant", age=10), "뿌우~"),
]

ISP_AGEABLE_CASES = [
    (AnimalData(name="Baby", species="lion", age=1), False),
    (AnimalData(name="Adult", species="lion", age=5), True),
]


# ============================================================================
# Dependency Inversion Principle (DIP) 테스트 데이터
# ============================================================================

DIP_ZOO_CONFIGS = [
    ZooConfig(name="Seoul Zoo", capacity=200),
    ZooConfig(name="Everland Zoo", capacity=300),
]

DIP_REPOSITORY_TEST_ANIMALS = [
    AnimalData(name="Tiger", species="lion", age=4),
    AnimalData(name="Elephant", species="elephant", age=15),
    AnimalData(name="Monkey", species="monkey", age=5),
]
