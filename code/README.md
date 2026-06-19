# Damage Claim Evidence Review — Code

Multi-modal pipeline that reads `dataset/claims.csv` (44 rows) and produces
`output.csv` with 14 structured columns per row, using Qwen3.5-Plus vision
via Alibaba Cloud DashScope US endpoint.

---

## Prerequisites

### Python

Python 3.10 or higher. A virtual environment is strongly recommended.

```powershell
# From the repo root (one level above challenge/)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Then from challenge/
pip install -r code/requirements.txt
```

### FFmpeg

Required to decode AVIF images submitted as `.jpg` files.
8 of the 82 image files in `claims.csv` (the final test set) are single-frame
ISOBMFF containers with `ftyp` major brand `avif`; 0 appear in the 29-image
sample set. Across both sets combined: 8 of 111 total files are AVIF.
FFmpeg 6+ is required; FFmpeg 8.x was used for the accepted inference run.

```powershell
ffmpeg -version   # must print a version line
```

Without FFmpeg, AVIF files yield zero frames and trigger the deterministic
`not_enough_information` / `evidence_standard_met=false` path with no API
call.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your DashScope API key:

```
DASHSCOPE_API_KEY=sk-your-actual-key-here
OPENAI_BASE_URL=https://dashscope-us.aliyuncs.com/compatible-mode/v1
```

**Region matching is mandatory.** The key created in the DashScope
International Console (US/Global zone) must be paired with the US endpoint
`https://dashscope-us.aliyuncs.com/compatible-mode/v1`. Keys created in the
China-mainland zone require the China endpoint and will not work here.

The US endpoint was used for all inference runs in this submission.

`.env` is gitignored and must never be committed.

Optional override:

```
VISION_MODEL=qwen3.5-plus   # default; use qwen3.5-flash for lower cost
```

---

## Commands

All commands run from `challenge/` with the virtual environment active.

### Offline checks (no API key needed)

```powershell
# Full test suite — 156 tests, no API calls
python -m pytest code/tests/ -v

# Dry-run import smoke test
python -m code.main --dry-run

# Verify media handling over both dataset CSVs
python -m pytest code/tests/test_media_audit.py -v -s
```

### Sample evaluation (20 rows from dataset/sample_claims.csv)

```powershell
# Infer on the 20-row labelled sample
python -m code.main --strategy strategy_a \
    --claims-csv dataset/sample_claims.csv \
    --output-csv output_strategy_a.csv

# Score against ground truth
python -m code.evaluation.main --strategy strategy_a \
    --ground-truth dataset/sample_claims.csv
```

### Full inference (44 rows from dataset/claims.csv)

```powershell
python -m code.main --strategy strategy_a
# Reads:  dataset/claims.csv
# Writes: output.csv  (gitignored)
```

Cache hits from previous runs are reused automatically. To force a complete
re-run, delete or rename `code/.cache/`.

### Structural validation of output.csv

```powershell
python docs/validate_output.py
```

Checks row count, column order, passthrough fields, enums, sentinel mixing,
supporting-ID references, and repeated-user-id preservation.

### Evaluation of all three strategies

```powershell
python -m code.evaluation.main --strategy all \
    --ground-truth dataset/sample_claims.csv
```

---

## Output Contract

`output.csv` always has exactly 14 columns in this order:

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | user_id | string | passthrough |
| 2 | image_paths | string | semicolon-separated; passthrough |
| 3 | user_claim | string | passthrough |
| 4 | claim_object | string | car / laptop / package; passthrough |
| 5 | evidence_standard_met | string | "true" / "false" |
| 6 | evidence_standard_met_reason | string | natural-language explanation |
| 7 | risk_flags | string | semicolon-sep; "none" if empty |
| 8 | issue_type | string | dent / scratch / crack / glass_shatter / ... |
| 9 | object_part | string | object-specific; front_bumper / screen / box / ... |
| 10 | claim_status | string | supported / contradicted / not_enough_information |
| 11 | claim_status_justification | string | natural-language explanation |
| 12 | supporting_image_ids | string | semicolon-sep; "none" if empty |
| 13 | valid_image | string | "true" / "false" |
| 14 | severity | string | none / low / medium / high / unknown |

---

## Architecture

```
claims.csv
    │
    ▼
pipeline.py          reads each row, checks cache, calls VLM
    │
    ├─► media.py     decodes image/video frames (FFmpeg for AVIF)
    │
    ├─► history.py   looks up user_history.csv → HistoryRecord
    │
    ├─► prompt.py    builds system + user messages (Strategy A/B/C)
    │
    ├─► vision_client.py  calls DashScope via OpenAI-compat SDK
    │
    ├─► cache.py     SHA-256-keyed JSON response cache
    │
    └─► validator.py deterministic post-processing
            │
            ├─ enum clamping
            ├─ history flag merge (user_history_risk, manual_review_required)
            ├─ supporting_image_ids subset filter
            ├─ contradicted-verdict consistency (evidence_standard_met=true)
            └─ valid_image=false safety (supported → NEI)
    │
    ▼
output.csv
```

### Strategy A (final selection)

Minimal prompt: claim text + image data only. No history summary, no evidence
rules, no calibration block. Clean verdicts with fewer context-induced
regressions. Selected after A/B/C comparison on the 20-row sample.

### Cache behavior

Responses are content-addressed by SHA-256 of the model name, strategy
version, claim row, and image bytes. Re-running with identical inputs produces
zero API calls. Cache directory: `code/.cache/` (gitignored).

### Retry behavior

Up to 3 attempts per row with exponential back-off. Each attempt is counted
separately in the run accounting summary. Cache misses after exhausting retries
fall through to the deterministic `not_enough_information` sentinel row.

### AVIF handling

`media.py` probes the first 12 bytes of each file for the `ftyp`+`avif` ISOBMFF
signature regardless of the `.jpg` extension. Detected AVIF files are decoded
via FFmpeg subprocess. Non-AVIF `.jpg` files are read directly with Pillow.

### Injection defense

The system prompt instructs the model to treat any text visible in images as
untrusted user content. The user message explicitly states that text found in
images is not authoritative. The pipeline never executes or evaluates strings
extracted from images.

---

## Cost Estimates

Prices are from the Alibaba Cloud DashScope International list as of 2026-06-19.
Free-tier usage may cost $0. Reported estimates reflect list-price calculations
and may not equal any amount actually charged.

| Model | Input / 1M tokens | Output / 1M tokens | Est. 44 rows |
|---|---|---|---|
| qwen3.5-plus | $0.40 | $2.40 | ~$0.046 |
| qwen3.5-flash | $0.07 | $0.30 | ~$0.008 |

The final accepted run used `qwen3.5-plus`: 68,218 input tokens + 7,763
output tokens over 44 rows (324.3 s, 0 retries).

---

## Packaging

Use the allowlist-based packaging script to build `code.zip`:

```powershell
# From challenge/
.\scripts\package.ps1
```

The script creates `code.zip` with a top-level `code/` directory containing
source, tests, prompts, evaluation, requirements, and documentation.
Excluded: `.env`, caches, `__pycache__`, logs, smoke outputs, `output.csv`,
and unrelated strategy artifacts.

`code.zip` is gitignored.

### Submission artifacts

Submit three files separately to HackerRank — do not bundle them together:

1. **`code.zip`** — source code archive (built by `scripts/package.ps1`)
2. **`output.csv`** — inference results for `claims.csv` (gitignored, never bundled)
3. **`chat_transcript.txt`** — AI interaction transcript (kept outside `code.zip`)
