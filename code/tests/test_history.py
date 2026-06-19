"""Tests for HistoryLoader."""
import pytest
from code.agent.history import HistoryLoader


def test_loader_reads_csv(repo_root):
    csv_path = repo_root / "dataset" / "user_history.csv"
    if not csv_path.exists():
        pytest.skip("user_history.csv not available")
    loader = HistoryLoader(csv_path)
    # At least one record must be loaded
    assert len(loader._records) > 0


def test_loader_get_existing(repo_root):
    csv_path = repo_root / "dataset" / "user_history.csv"
    if not csv_path.exists():
        pytest.skip("user_history.csv not available")
    loader = HistoryLoader(csv_path)
    # user_001 is in the dataset
    rec = loader.get("user_001")
    assert rec is not None
    assert rec.user_id == "user_001"
    assert isinstance(rec.past_claim_count, int)


def test_loader_get_missing(repo_root):
    csv_path = repo_root / "dataset" / "user_history.csv"
    if not csv_path.exists():
        pytest.skip("user_history.csv not available")
    loader = HistoryLoader(csv_path)
    assert loader.get("nonexistent_user_xyz") is None
