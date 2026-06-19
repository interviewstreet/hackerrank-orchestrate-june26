# Damage Claim Evidence Review — Code

Multi-modal pipeline that reads `dataset/claims.csv` (44 rows) and produces
`output.csv` with 14 structured columns per row, using Qwen3.5 vision via
Alibaba Cloud DashScope International.

---

## Setup

### 1. Python environment

```powershell
# From repo root (D:\HackerRank\orchestrate-june-2026\)
.\.venv\Scripts\Activate.ps1
pip install -r code/requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and add your DashScope API key:

```
DASHSCOPE_API_KEY=sk-your-actual-key
```

`.env` is gitignored and must never be committed.

### 3. FFmpeg (required for video files)

8 of the 111 submitted files are MP4 videos disguised as `.jpg`.
FFmpeg 8.x is required to extract frames from them.

```powershell
# Verify
ffmpeg -version
```

Without FFmpeg, video files return zero frames and take the
`not_enough_information` deterministic path (no API call).

---

## Commands

All commands run from `challenge/` with the venv active.

### Offline checks (no API key needed)

```powershell
# Run 62 offline tests
python -m pytest code/tests/ -v

# Import smoke test (no API calls)
python -m code.main --dry-run

# Media audit over both CSV files
python -m pytest code/tests/test_media_audit.py -v -s
```

### Sample evaluation (20 rows, costs ~$0.02 per strategy)

```powershell
# Strategy A (minimal prompt — baseline)
python -m code.main --strategy strategy_a `
    --claims-csv dataset/sample_claims.csv `
    --output-csv output_strategy_a.csv

# Score against sample ground truth
python -m code.evaluation.main --strategy strategy_a

# Strategy B (context-rich — candidate)
python -m code.main --strategy strategy_b `
    --claims-csv dataset/sample_claims.csv `
    --output-csv output_strategy_b.csv

python -m code.evaluation.main --strategy strategy_b
```

### Full inference (44 rows, costs ~$0.05 per strategy)

```powershell
python -m code.main --strategy strategy_b
# Writes: challenge/output.csv
```

---

## Output columns

`output.csv` always has exactly 14 columns in this order:

| Column | Type | Notes |
|---|---|---|
| user_id | string | |
| image_paths | string | semicolon-separated, as-is from input |
| user_claim | string | |
| claim_object | string | car / laptop / package |
| evidence_standard_met | string | "true" / "false" |
| evidence_standard_met_reason | string | |
| risk_flags | string | semicolon-sep; "none" if empty |
| issue_type | string | see allowed list |
| object_part | string | object-specific; see allowed list |
| claim_status | string | supported / contradicted / not_enough_information |
| claim_status_justification | string | |
| supporting_image_ids | string | semicolon-sep; "none" if empty |
| valid_image | string | "true" / "false" |
| severity | string | none / low / medium / high / unknown |

---

## Cost estimates

Prices are from the Alibaba Cloud DashScope International list as of 2026-06-19.
Free-tier usage may cost $0. Reported estimates reflect list-price calculations
and may not equal any amount actually charged.

| Model | Input / 1M tokens | Output / 1M tokens | Est. 44 rows |
|---|---|---|---|
| qwen3.5-plus | $0.40 | $2.40 | ~$0.05 |
| qwen3.5-flash | $0.07 | $0.30 | ~$0.01 |

---

## Cache

Responses are stored in `code/.cache/` (gitignored, SHA256-keyed JSON).
Re-running with the same inputs hits the cache and makes zero API calls.
Delete `code/.cache/` to force a full re-run.

---

## Packaging

```powershell
# From challenge/
Compress-Archive -Path code/ -DestinationPath code.zip `
    -CompressionLevel Optimal
# Verify size
(Get-Item code.zip).Length / 1MB
```

`code.zip` is gitignored. Do not include `.env`, `.cache/`, or `__pycache__`.
