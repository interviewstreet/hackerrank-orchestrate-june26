"""Entry point: python -m code.main [OPTIONS]

Reads challenge/dataset/claims.csv, runs the pipeline for every row,
and writes challenge/output.csv (44 rows, 14 columns).

Usage:
    python -m code.main                        # Strategy B (default)
    python -m code.main --strategy strategy_a  # baseline
    python -m code.main --strategy strategy_b  # context-rich (default)
    python -m code.main --dry-run              # import check; no API calls
    python -m code.main --limit 5              # process only first N rows
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure challenge/ is on sys.path so `code.*` imports work when run from
# inside the challenge/ directory (python -m code.main).
_CHALLENGE_ROOT = Path(__file__).parent.parent
if str(_CHALLENGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE_ROOT))


def _load_env() -> None:
    """Load .env from challenge/ root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_path = _CHALLENGE_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def main(argv: list[str] | None = None) -> int:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Run the damage-claim evidence review pipeline."
    )
    parser.add_argument(
        "--strategy",
        choices=["strategy_a", "strategy_b"],
        default="strategy_b",
        help="Prompt strategy to use (default: strategy_b)",
    )
    parser.add_argument(
        "--claims-csv",
        default=str(_CHALLENGE_ROOT / "dataset" / "claims.csv"),
        help="Path to input claims.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=str(_CHALLENGE_ROOT / "output.csv"),
        help="Path to write output.csv",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_CHALLENGE_ROOT / "code" / ".cache"),
        help="Path to disk cache directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import all modules and print config; do not call the API",
    )
    args = parser.parse_args(argv)

    # Import here so --help works even without a venv
    from code.agent.accounting import RunAccounting
    from code.agent.cache import CacheStore
    from code.agent.evidence import EvidenceLoader
    from code.agent.history import HistoryLoader
    from code.agent.models import ClaimRow
    from code.agent.pipeline import run_row

    claims_path = Path(args.claims_csv)
    output_path = Path(args.output_csv)
    cache_dir = Path(args.cache_dir)
    strategy = args.strategy

    if not claims_path.exists():
        print(f"ERROR: claims CSV not found: {claims_path}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run: all imports successful.")
        print(f"  claims_csv : {claims_path}")
        print(f"  output_csv : {output_path}")
        print(f"  cache_dir  : {cache_dir}")
        print(f"  strategy   : {strategy}")
        print(f"  provider   : {os.environ.get('MODEL_PROVIDER', 'qwen')}")
        print(f"  model      : {os.environ.get('VISION_MODEL', 'qwen3.5-plus')}")
        return 0

    history_loader = HistoryLoader(
        _CHALLENGE_ROOT / "dataset" / "user_history.csv"
    )
    evidence_loader = EvidenceLoader(
        _CHALLENGE_ROOT / "dataset" / "evidence_requirements.csv"
    )
    cache = CacheStore(cache_dir)

    # Lazy-init the VLM client only when actually needed (not dry-run)
    from code.agent.vision_client import get_client
    client = get_client()

    accounting = RunAccounting()

    with open(claims_path, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]

    output_rows = []
    for i, row_dict in enumerate(rows, 1):
        claim = ClaimRow(
            user_id=row_dict["user_id"],
            image_paths=row_dict["image_paths"],
            user_claim=row_dict["user_claim"],
            claim_object=row_dict["claim_object"],
        )
        try:
            output_row, stats = run_row(
                claim=claim,
                repo_root=_CHALLENGE_ROOT,
                history_loader=history_loader,
                evidence_loader=evidence_loader,
                cache=cache,
                client=client,
                strategy=strategy,
            )
            accounting.add(stats)
            output_rows.append(output_row)
            status_icon = "C" if stats.cache_hit else "."
            error_tag = f" [ERROR: {stats.error[:40]}]" if stats.error else ""
            print(f"[{i:02d}/{len(rows)}] {status_icon} {claim.user_id}{error_tag}")
        except Exception as exc:
            print(f"[{i:02d}/{len(rows)}] FAIL {claim.user_id}: {exc}", file=sys.stderr)
            from code.agent.validator import zero_media_output
            history = history_loader.get(claim.user_id)
            output_rows.append(zero_media_output(claim, history))

    # Write output.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row.model_dump())

    print(f"\nWrote {len(output_rows)} rows to {output_path}")
    print(accounting.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
