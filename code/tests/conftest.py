import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent  # challenge/


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def sample_csv(repo_root: Path) -> Path:
    return repo_root / "dataset" / "sample_claims.csv"


@pytest.fixture
def claims_csv(repo_root: Path) -> Path:
    return repo_root / "dataset" / "claims.csv"


@pytest.fixture
def images_root(repo_root: Path) -> Path:
    return repo_root / "dataset" / "images"
