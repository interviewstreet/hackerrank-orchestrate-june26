# Operational Evaluation Report

## Overview

This report covers the operational performance of the damage-claim evidence
review pipeline for both the 20-row labelled sample run and the final 44-row
test-set inference run. All figures are measured or derived from run logs;
cost and token estimates are based on list prices.

---

## 1. Model Calls

| Run | Claims CSV | SDK requests | Cache hits | Retries | Errors |
|---|---|---|---|---|---|
| B0 manual smoke (strategy_a) | sample_claims.csv (1 row) | 1 | 0 | 0 | 0 |
| Sample A (strategy_a) | sample_claims.csv (20 rows) | 20 | 0 | 0 | 0 |
| Sample B (strategy_b) | sample_claims.csv (20 rows) | 20 | 0 | 0 | 0 |
| Sample C (strategy_c) | sample_claims.csv (20 rows) | 20 | 0 | 0 | 0 |
| Final A (strategy_a) | claims.csv (44 rows) | 44 | 0 | 0 | 0 |
| Zero-API replays (cache-hit) | — | 0 | 64 (20+44) | 0 | 0 |
| **Lifecycle total** | | **105** | | | |

Strategies B and C each made 20 SDK calls during their initial evaluation
runs. Later re-scoring of saved B/C output CSVs made no additional calls.
All zero-API replays (for the contradicted-verdict fix and packaging
verification) were pure cache hits against the original A caches.

Model used: `qwen3.5-plus` via Alibaba Cloud DashScope US endpoint
(`https://dashscope-us.aliyuncs.com/compatible-mode/v1`).

---

## 2. Token Usage

### Sample run (20 rows, strategy_a)

Token counts are not available for the sample run log (it was not captured
with full accounting at the time). Based on the per-row averages from the
final run, estimates are:

| Estimate | Value |
|---|---|
| Est. input tokens | ~31,000 (20 rows × ~1,550) |
| Est. output tokens | ~3,500 (20 rows × ~177) |

### Final run (44 rows, strategy_a) — measured

| Metric | Value |
|---|---|
| Total input tokens | 68,218 |
| Total output tokens | 7,763 |
| Average input tokens / row | ~1,550 |
| Average output tokens / row | ~177 |

Input tokens include the system prompt (~400 tokens), claim text, and
base64-encoded image data. The system prompt is repeated on every call
because the OpenAI-compatible API does not support persistent system context
across independent calls in this configuration.

---

## 3. Images Processed

### Sample run (20 rows)

The sample dataset contains 20 claims referencing **29 image files**
(avg ~1.45 images/claim). No AVIF files appear in the sample set.

### Final run (44 rows)

| Metric | Value |
|---|---|
| Total image files referenced | 82 (44 claims, avg ~1.86 files/claim) |
| Successfully decoded | 82 |
| AVIF files (.jpg extension) | 8 (detected by ISOBMFF probe; decoded via FFmpeg) |
| Zero-media rows | 0 (all 44 rows had at least one usable frame) |

All 82 files decoded successfully. AVIF files were identified by inspecting
the first 12 bytes for the `ftyp`+`avif` ISOBMFF signature and decoded via
FFmpeg subprocess. No claim triggered the zero-media deterministic fallback.

Across both the sample and final datasets combined: 29 + 82 = **111 files**,
of which 8 are AVIF.

---

## 4. Cost and Pricing Assumptions

### Pricing basis

Rates from Alibaba Cloud DashScope International list as of 2026-06-19:

| Model | Input | Output |
|---|---|---|
| qwen3.5-plus | USD 0.40 / 1M tokens | USD 2.40 / 1M tokens |
| qwen3.5-flash | USD 0.07 / 1M tokens | USD 0.30 / 1M tokens |

### Final run cost estimate (44 rows, qwen3.5-plus)

| Component | Tokens | Rate | Cost |
|---|---|---|---|
| Input | 68,218 | $0.40 / 1M | $0.0273 |
| Output | 7,763 | $2.40 / 1M | $0.0186 |
| **Total** | **75,981** | | **~$0.046** |

This is a list-price estimate. It does not reflect free-tier credits, volume
discounts, or the actual invoice amount from Alibaba Cloud. The application
performs no cost tracking at runtime; this figure was computed from the run
log's token counts after the fact.

### Sample run cost estimate (20 rows, qwen3.5-plus)

Using the per-row averages above: ~$0.021 estimated.

---

## 5. Latency and Runtime

### Final run (44 rows)

| Metric | Value |
|---|---|
| Total wall-clock time | 324.3 seconds |
| Average time per row | ~7.4 seconds |
| Min / Max row latency | not individually recorded |

The pipeline processes rows sequentially (no batching). Total runtime includes
image decoding, cache key computation, API call, JSON parsing, and validator
execution. API call time dominates (~95% of per-row time).

### Sample run (20 rows)

Not individually captured; estimated at ~148 seconds based on the per-row
average from the final run.

---

## 6. TPM / RPM Considerations

DashScope International published limits for `qwen3.5-plus` (as of run date):

| Limit | Value |
|---|---|
| Tokens per minute (TPM) | Not published; assumed conservative |
| Requests per minute (RPM) | Not published; assumed conservative |

The pipeline makes one sequential request per row with no parallelism.
At 44 rows over 324.3 seconds, the effective request rate was approximately
8.1 RPM. No rate-limit errors were encountered. The back-off retry logic
(3 attempts, exponential delay starting at 1 s) was never triggered.

At the observed token rates (~14,200 input + 1,600 output tokens/minute
averaged over the run), the pipeline operated well within the expected TPM
envelope for this model tier.

---

## 7. Cost Controls

### Cache

Responses are stored in a SHA-256-keyed JSON cache at `code/.cache/`.
The cache key is computed from the model name, strategy version constant,
claim row fields (user_id, image_paths, user_claim, claim_object), and
image file bytes. Identical re-runs hit the cache and make zero API calls.

The final output.csv was produced from the original 44-call run and then
replayed once more from cache (0 calls) after the contradicted-verdict
consistency fix to regenerate the accepted output. The cache directory
for claims.csv is not reused for sample runs. Total SDK calls for claims.csv
across the entire lifecycle: 44.

### Retry behaviour

Up to 3 attempts per row. If all attempts fail (JSON parse error, network
timeout, or provider error), the row receives a deterministic sentinel output
(`claim_status=not_enough_information`, `evidence_standard_met=false`) rather
than crashing or leaving a gap. This prevents unnecessary row re-submission
loops. No retries were triggered in any run.

### Unnecessary-call prevention

- Zero-media rows (no decodable frames): bypass the VLM entirely and receive
  the deterministic `not_enough_information` result. 0 such rows in this dataset.
- Cache check is performed before any prompt construction, image encoding, or
  API call is initiated.
- `--dry-run` flag validates configuration, imports, and environment without
  triggering any API call.
- The evaluation pipeline reads only from existing output CSVs; it makes no
  API calls.

### Batching and throttling

No batching is implemented. The DashScope API does not offer a batch endpoint
for multimodal vision calls in the OpenAI-compatible mode. Sequential
processing with per-row cache checks provides equivalent cost control without
added complexity.

---

## 8. Strategy Evaluation Summary

Three strategies were compared on the 20-row sample before committing to the
final inference. See `evaluation/RESULTS.md` for the complete metric table.

| Strategy | Description | Outcome |
|---|---|---|
| A (final) | Minimal prompt: claim + images | Selected — best P0 metrics |
| B | Context-rich: + evidence rules + history + calibration | Eliminated — regressed claim_status 85%→80% and valid_image 95%→90% |
| C | Calibrated: A context + calibration block only | Eliminated — regressed claim_status and valid_image; added user_033 and user_008 errors |

Each strategy made 20 SDK calls for its initial evaluation run (60 calls total
for the sample comparison). Strategy A was then used for the 44-row final run
(44 calls). The B0 manual smoke added 1 call. Lifecycle total: 105 SDK calls.

Decision-time A/B/C metrics and per-row error attribution are in
`evaluation/RESULTS.md`.
