# Evaluation Results — Strategy A (Final)

## Strategy Selection

Three strategies were evaluated on the 20-row labelled sample before the final
44-row inference run. All strategies use the same deterministic validator.

| Strategy | Description |
|---|---|
| A | Minimal prompt: claim text + images only |
| B | Context-rich: A + evidence rules + user history summary + calibration guidance |
| C | Calibrated baseline: A context + shared calibration block (no history, no evidence) |

**Winner: Strategy A**

Strategy A had three claim-status misses (user_005, user_020, user_034; all
gold `contradicted`, all predicted `supported`), giving 85.0% claim_status
accuracy.

Strategy B fixed user_020 (correctly predicted `contradicted`) but introduced
new errors on user_032 and user_033, so claim_status fell to 80.0%. B also
regressed valid_image accuracy from 95.0% to 90.0%.

Strategy C retained the user_020 miss, introduced the user_033 regression, and
additionally regressed valid_image on user_008 (from 95.0% to 90.0%).

P0 priority was verdict accuracy (claim_status) and valid_image. Both B and C
regressed on these P0 fields, disqualifying them despite P1/P2 gains.

---

## Final Sample Metrics (Strategy A, post-validator)

Measured against the 20-row labelled sample (`dataset/sample_claims.csv`).
All values below are exact counts or micro-averaged; no estimation or rounding
beyond normal floating-point display.

### Exact-Match Accuracy

| Field | Correct / Total | Accuracy |
|---|---|---|
| evidence_standard_met | 20 / 20 | **100.0%** |
| valid_image | 19 / 20 | 95.0% |
| claim_status | 17 / 20 | 85.0% |
| object_part | 17 / 20 | 85.0% |
| issue_type | 10 / 20 | 50.0% |
| severity | 9 / 20 | 45.0% |

### Set Micro-F1

| Field | Precision | Recall | F1 |
|---|---|---|---|
| risk_flags | 86.1% | 83.8% | **84.9%** |
| supporting_image_ids | 95.2% | 95.2% | **95.2%** |

### Claim-Status Confusion Matrix

|  | pred: supported | pred: contradicted | pred: NEI |
|---|---|---|---|
| gold: supported | 13 | 0 | 0 |
| gold: contradicted | 3 | 2 | 0 |
| gold: NEI | 0 | 0 | 2 |

Strategy A's three misses: user_005, user_020, and user_034. All three are
gold `contradicted` and were predicted `supported` by Strategy A.

---

## Strategy Comparison (sample, decision-time metrics)

Metrics captured at the time the A/B/C decision was made (before the
contradicted-verdict validator repair). Each strategy made 20 SDK calls for
its initial evaluation run.

| Metric | A | B | C |
|---|---|---|---|
| evidence_standard_met | 90.0% | 90.0% | 90.0% |
| valid_image | **95.0%** | 90.0% | 90.0% |
| claim_status | **85.0%** | 80.0% | 80.0% |
| object_part | 85.0% | 80.0% | **90.0%** |
| issue_type | 50.0% | **65.0%** | 55.0% |
| severity | 45.0% | **60.0%** | 55.0% |
| risk_flags F1 | 84.9% | 86.1% | **91.4%** |
| supporting_image_ids F1 | **85.7%** | 85.7% | 81.8% |

After the deterministic contradicted-verdict repair (zero additional API
calls), Strategy A's sample metrics improved: evidence_standard_met 90%→**100%**
and supporting_image_ids F1 85.7%→**95.2%**. All other A metrics are unchanged.

---

## Final 44-Row Inference Accounting

Run date: 2026-06-19/20.
Cache directory: `code/.cache/claims_strategy_a_final_v1` (44 entries, gitignored).

| Metric | Value |
|---|---|
| Rows processed | 44 |
| Zero-media rows | 0 |
| SDK requests | 44 |
| Retries | 0 |
| Errors | 0 |
| Total input tokens | 68,218 |
| Total output tokens | 7,763 |
| Avg input tokens / row | ~1,550 |
| Total latency | 324.3 s |
| Est. list-price cost | ~USD 0.046 |

The cost figure is a list-price estimate computed from the provider's published
rates at the time of the run. It does not reflect any actual charge, discount,
or free-tier credit and may differ from the invoice amount.

---

## Final Output Distribution (44 rows)

| Field | Value | Count |
|---|---|---|
| claim_status | supported | 21 |
| claim_status | contradicted | 14 |
| claim_status | not_enough_information | 9 |
| valid_image | true | 43 |
| valid_image | false | 1 |
| evidence_standard_met | true | 35 |
| evidence_standard_met | false | 9 |

The 9 NEI rows all have `evidence_standard_met=false`.
All 14 contradicted rows have `evidence_standard_met=true` (enforced by
the deterministic post-validator).
The 21 supported rows have `evidence_standard_met` set by the model (all
true in the accepted output).

---

## Test Coverage

```
156 passed, 0 failed
```

Test modules: cache, evaluation/metrics, evidence, history, media, media_audit,
models, output, pipeline, prompt, validator (including 6 contradicted-consistency
tests added in the final repair), vision_client_bool.
