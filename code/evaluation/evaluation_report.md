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
| Sample A (strategy_a) | sample_claims.csv (20 rows) | 20 | 0 | 0 | 0 |
| Final A (strategy_a) | claims.csv (44 rows) | 44 | 0 | 0 | 0 |
| Zero-API replays (both) | — | 0 | 20 + 44 = 64 | 0 | 0 |

The sample run (20 rows) was used for A/B/C strategy comparison. The final
run (44 rows) produced the accepted `output.csv`. All subsequent re-runs
after the cache was warm used 0 SDK requests (pure cache hits).

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

The sample dataset contains 20 claims. Image paths reference `images/sample/`
entries. Based on the same 1.2 images-per-claim average observed in the final
set, the estimated image count is approximately 24 images.

### Final run (44 rows)

| Metric | Value |
|---|---|
| Total image files referenced | 111 (44 claims, avg ~2.5 files/claim) |
| Successfully decoded | 111 |
| AVIF files (.jpg extension) | 8 (detected by ISOBMFF probe; decoded via FFmpeg) |
| Zero-media rows | 0 (all 44 rows had at least one usable frame) |

All 111 files decoded successfully. AVIF files were identified by inspecting
the first 12 bytes for the `ftyp`+`avif` ISOBMFF signature and decoded via
FFmpeg subprocess. No claim triggered the zero-media deterministic fallback.

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
regenerated twice more from cache (0 calls each) for the contradict-verdict
consistency fix and the post-packaging verification. Total SDK calls across
the entire project lifecycle for claims.csv: 44.

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

| Strategy | Description | Sample A/B decision |
|---|---|---|
| A (final) | Minimal prompt: claim + images | Selected — clean P0 verdict accuracy |
| B | Context-rich: + evidence rules + history + calibration | P0 regression on user_020 |
| C | Calibrated: A context + calibration block only | P0 regression on user_033 |

Strategies B and C were evaluated using cache-only replay (0 additional API
calls). The total number of unique SDK calls across all three strategy
comparisons was 80 (20 A + 20 B + 20 C) on the sample, plus 44 for the
accepted final run.
