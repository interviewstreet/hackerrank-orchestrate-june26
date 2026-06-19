"""Tests for run_row pipeline with a mocked VisionClient."""
import io
import json
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock

from code.agent.cache import CacheStore
from code.agent.evidence import EvidenceLoader
from code.agent.history import HistoryLoader
from code.agent.models import ClaimRow, MediaFile, ModelOutput, RowStats
from code.agent.pipeline import run_row
from code.agent.prompt import STRATEGY_B
from code.agent.vision_client import VisionClient


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 150, 100)).save(buf, "JPEG")
    return buf.getvalue()


def _mock_client(output: ModelOutput) -> VisionClient:
    client = MagicMock(spec=VisionClient)
    client.provider = "qwen"
    client.model = "qwen3.5-plus"

    def fake_call(system_prompt, user_content, stats):
        stats.input_tokens = 500
        stats.output_tokens = 150
        stats.latency_ms = 200.0
        stats.provider = "qwen"
        stats.model = "qwen3.5-plus"
        return output

    client.call.side_effect = fake_call
    return client


@pytest.fixture
def sample_claim(repo_root):
    return ClaimRow(
        user_id="user_002",
        image_paths="dataset/images/sample/case_001/img_1.jpg",
        user_claim="My car door is dented.",
        claim_object="car",
    )


@pytest.fixture
def model_output():
    return ModelOutput(
        evidence_standard_met=True,
        evidence_standard_met_reason="Dent clearly visible.",
        risk_flags=["none"],
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="img_1 shows door dent.",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity="medium",
    )


def test_run_row_zero_media(repo_root, sample_claim, model_output, tmp_path):
    """When all image paths are missing, short-circuit to not_enough_information."""
    claim = ClaimRow(
        user_id="user_002",
        image_paths="images/test/case_NONEXISTENT/img_99.jpg",
        user_claim="My car door is dented.",
        claim_object="car",
    )
    history_loader = HistoryLoader(repo_root / "dataset" / "user_history.csv")
    evidence_loader = EvidenceLoader(repo_root / "dataset" / "evidence_requirements.csv")
    cache = CacheStore(tmp_path)
    client = _mock_client(model_output)

    output_row, stats = run_row(
        claim=claim,
        repo_root=repo_root,
        history_loader=history_loader,
        evidence_loader=evidence_loader,
        cache=cache,
        client=client,
        strategy=STRATEGY_B,
    )
    assert output_row.claim_status == "not_enough_information"
    assert output_row.valid_image == "false"
    client.call.assert_not_called()  # no API call for zero-media


def test_run_row_cache_miss_calls_client(repo_root, sample_claim, model_output, tmp_path):
    """On cache miss, client.call() is invoked and result is cached."""
    history_loader = HistoryLoader(repo_root / "dataset" / "user_history.csv")
    evidence_loader = EvidenceLoader(repo_root / "dataset" / "evidence_requirements.csv")
    cache = CacheStore(tmp_path)
    client = _mock_client(model_output)

    img_path = repo_root / "dataset" / "images" / "sample" / "case_001" / "img_1.jpg"
    if not img_path.exists():
        pytest.skip("Sample image not available")

    output_row, stats = run_row(
        claim=sample_claim,
        repo_root=repo_root,
        history_loader=history_loader,
        evidence_loader=evidence_loader,
        cache=cache,
        client=client,
        strategy=STRATEGY_B,
    )
    assert client.call.called
    assert stats.cache_hit is False
    assert output_row.user_id == "user_002"


def test_run_row_cache_hit_skips_client(repo_root, sample_claim, model_output, tmp_path):
    """Second call with same inputs must be served from cache (no API)."""
    history_loader = HistoryLoader(repo_root / "dataset" / "user_history.csv")
    evidence_loader = EvidenceLoader(repo_root / "dataset" / "evidence_requirements.csv")
    cache = CacheStore(tmp_path)
    client = _mock_client(model_output)

    img_path = repo_root / "dataset" / "images" / "sample" / "case_001" / "img_1.jpg"
    if not img_path.exists():
        pytest.skip("Sample image not available")

    # First call
    run_row(
        claim=sample_claim, repo_root=repo_root,
        history_loader=history_loader, evidence_loader=evidence_loader,
        cache=cache, client=client, strategy=STRATEGY_B,
    )
    assert client.call.call_count == 1

    # Second call — must be cache hit
    _, stats2 = run_row(
        claim=sample_claim, repo_root=repo_root,
        history_loader=history_loader, evidence_loader=evidence_loader,
        cache=cache, client=client, strategy=STRATEGY_B,
    )
    assert client.call.call_count == 1  # still 1, not 2
    assert stats2.cache_hit is True
