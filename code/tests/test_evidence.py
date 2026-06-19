"""Tests for EvidenceLoader."""
import pytest
from code.agent.evidence import EvidenceLoader


def test_loads_rules(repo_root):
    csv_path = repo_root / "dataset" / "evidence_requirements.csv"
    if not csv_path.exists():
        pytest.skip("evidence_requirements.csv not available")
    loader = EvidenceLoader(csv_path)
    assert len(loader.rules) > 0


def test_get_all_for_car(repo_root):
    csv_path = repo_root / "dataset" / "evidence_requirements.csv"
    if not csv_path.exists():
        pytest.skip("evidence_requirements.csv not available")
    loader = EvidenceLoader(csv_path)
    rules = loader.get_all_for_object("car")
    assert len(rules) > 0
    # All returned rules apply to 'car' or 'all'
    for r in rules:
        assert r.claim_object in ("car", "all")


def test_format_for_prompt(repo_root):
    csv_path = repo_root / "dataset" / "evidence_requirements.csv"
    if not csv_path.exists():
        pytest.skip("evidence_requirements.csv not available")
    loader = EvidenceLoader(csv_path)
    text = loader.format_for_prompt("laptop")
    assert "[" in text
    assert "(" in text


def test_unknown_object_returns_only_all_rules(repo_root):
    csv_path = repo_root / "dataset" / "evidence_requirements.csv"
    if not csv_path.exists():
        pytest.skip("evidence_requirements.csv not available")
    loader = EvidenceLoader(csv_path)
    rules = loader.get_all_for_object("spaceship")
    for r in rules:
        assert r.claim_object == "all"
