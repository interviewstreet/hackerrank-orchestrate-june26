# Judge Walkthrough — Damage Claim Evidence Review

## Problem Framing

The task is to build an automated agent that verifies whether submitted image
evidence supports, contradicts, or is insufficient to adjudicate a damage
insurance claim. For each of the 44 rows in `claims.csv`, the pipeline must:

1. Decode all image evidence (including disguised AVIF files).
2. Optionally enrich context from the user's claim history.
3. Call a vision-language model to review the evidence against the claim.
4. Apply deterministic post-processing to enforce output contracts.
5. Write one output row per claim with exactly 14 structured columns.

The challenge requires the output to be reproducible, structurally valid, and
semantically coherent with the labelled sample contract.

---

## Pipeline Stages

### Stage 0: Input loading

`pipeline.py` iterates `claims.csv` row by row. For each row it:
- Reads passthrough columns (user_id, image_paths, user_claim, claim_object).
- Builds a cache key from all inputs (model, strategy, version constant, claim
  row fields, image bytes) using SHA-256.

### Stage 1: Media decoding (`media.py`)

Each image path is read and probed for AVIF signatures. Files with `ftyp`+`avif`
ISOBMFF headers (detectable in the first 12 bytes) are decoded via FFmpeg
subprocess even when the extension is `.jpg`. Other images are read with Pillow
and resized to a maximum long-edge of 1024 px before base64-encoding.

8 of the 82 files in the final test set (claims.csv) are AVIF files with
`.jpg` extensions. All 8 decoded successfully. Rows with zero usable frames
take a deterministic `not_enough_information` path and make no API call.
(The combined sample + final dataset totals 111 files; 0 AVIF appear in the
sample set.)

### Stage 2: User history lookup (`history.py`)

`dataset/user_history.csv` is loaded once. For each claim's user_id, a
`HistoryRecord` is built with claim counts, accept/reject ratios, and the
literal `history_flags` string from the dataset (e.g. `user_history_risk`,
`manual_review_required`, or `none`).

In Strategy A, history is used only by the deterministic validator — not passed
to the model.

### Stage 3: Prompt construction (`prompt.py`)

**Strategy A** (final selection): the user message contains only the claim text
and image data. The system prompt includes a JSON schema, output-contract rules,
and an injection-guard clause instructing the model to treat text visible in
images as untrusted user content.

**Strategy B**: adds evidence rules, user history summary, and calibration
guidance. Fixed the user_020 miss but introduced new claim-status errors on
user_032 and user_033, and regressed valid_image on user_020. Net P0 loss
eliminated B.

**Strategy C**: Strategy A context + calibration block only (no history or
evidence rules). Retained the user_020 miss, introduced the user_033 regression
(wrong-object contradiction became NEI), and regressed valid_image on user_008.
Eliminated.

### Stage 4: VLM call (`vision_client.py`)

Calls the model via the OpenAI-compatible DashScope US endpoint. The response
is a structured JSON object matching the `ModelOutput` Pydantic schema. If JSON
parsing fails, the row retries up to 3 times with exponential back-off.

### Stage 5: Deterministic post-processing (`validator.py`)

Applied to every row including cache hits. Enforces:

1. **Enum clamping**: invalid strings default to safe values
   (claim_status → not_enough_information, severity → unknown, etc.).
2. **History flag merge**: if `user_history_risk` (or similar trigger) is in
   the history flags, `user_history_risk` and `manual_review_required` are
   added to `risk_flags`. The literal string `"user_history_risk"` is the value
   used in the dataset and is recognised by the merge logic.
3. **supporting_image_ids subset filter**: any ID not in the set of submitted
   image file stems is removed; the sentinel `"none"` is substituted if all
   are stripped.
4. **Contradicted-verdict consistency**: when `claim_status == "contradicted"`,
   `evidence_standard_met` is forced to `true` regardless of the model's
   response, and supporting IDs are preserved or fallen back to the full
   submitted set. Rationale: a definite contradiction proves the evidence was
   sufficient to reach a verdict.
5. **valid_image=False safety**: for non-contradicted rows, if the model flags
   no valid image, `supported` is downgraded to `not_enough_information` and
   evidence fields are cleared.

---

## Strategy A/B/C Decision

### Evaluation setup

All strategies were evaluated on the 20-row labelled sample
(`dataset/sample_claims.csv`). Each strategy made 20 SDK calls for its initial
evaluation run (60 total). No strategy re-used another's cache.

### Priority tiers

- **P0 (critical)**: claim_status accuracy, valid_image accuracy
- **P1 (important)**: evidence_standard_met, object_part, risk_flags F1, supporting_image_ids F1
- **P2 (secondary)**: issue_type, severity

### Why A won over B

Strategy A had three claim-status misses (user_005, user_020, user_034; all
gold `contradicted`, predicted `supported`), giving 85.0% claim_status accuracy.
Strategy B fixed user_020 but introduced new errors on user_032 and user_033,
and regressed valid_image from 95.0% to 90.0%, so claim_status fell to 80.0%.
Despite B's gains in issue_type (+15%), severity (+15%), and risk_flags F1, the
P0 losses disqualified it.

### Why A won over C

Strategy C retained Strategy A's user_020 miss, introduced the user_033
regression (wrong-object contradiction became NEI), and additionally regressed
valid_image on user_008, also dropping valid_image from 95.0% to 90.0%. P0
losses on two fields eliminated C despite C's better object_part (90.0%) and
risk_flags F1 (91.4%).

---

## Edge Cases and How They Are Handled

| Edge case | Rows | Handling |
|---|---|---|
| AVIF images disguised as .jpg | 8 files across multiple rows | FFmpeg probe + decode |
| Text-injection attempts in images | 7 rows | Injection guard in system + user prompt |
| Non-original / stock images | 7 rows | `non_original_image` risk flag; valid_image=false where no original exists |
| valid_image=false + contradicted | 1 row (user_044) | Validator preserves contradicted verdict, sets evidence=true, falls back to submitted IDs |
| Multiple claims per row | 6 rows | Model handles per-row; returns primary object part |
| Repeated user_ids (multi-case) | 8 users appear 2–3x | Positional iteration preserves all rows |
| Possible manipulation | 1 row (user_044 iStock watermark) | `possible_manipulation` flag |
| History with no flags but non-zero claim count | Multiple rows | Only literal history_flags string drives merge; counts are ignored |

---

## Testing

156 offline tests covering all pipeline layers. No test makes API calls.

- `test_cache.py` — cache key isolation and version constants
- `test_evaluation.py` — metrics computation (exact-match and set F1)
- `test_evidence.py` — evidence module
- `test_history.py` — history flag parsing and HistoryRecord construction
- `test_media.py` / `test_media_audit.py` — AVIF detection and frame decoding
- `test_models.py` — Pydantic model validation and bool coercion
- `test_output.py` — OutputRow field contract
- `test_pipeline.py` — pipeline integration with mock VLM
- `test_prompt.py` — strategy A/B/C prompt content checks
- `test_validator.py` — enum enforcement, history merge, supporting_image_ids
  filter, contradicted-verdict consistency (6 dedicated tests)
- `test_vision_client_bool.py` — bool coercion edge cases

---

## Cost Controls

- All responses are cached by content-addressed SHA-256. Re-runs are free.
- The final 44-row run used 68,218 input tokens + 7,763 output tokens across
  44 SDK calls (0 retries, 0 errors) in 324.3 seconds, estimated at ~USD 0.046
  at list price.
- Budget alternative: set `VISION_MODEL=qwen3.5-flash` for ~8x lower cost at
  the risk of reduced reasoning quality.

---

## AI-Assisted Development Disclosure

This solution was built using Claude Code (Anthropic) as the primary code
author and Codex (OpenAI) as product manager, technical architect, and
code reviewer. The human engineer (HackerRank user zhuruizhi.zzz@gmail.com)
provided direction, reviewed all code and evaluation outputs, and made all
strategy and architecture decisions.

Specific contributions:
- **Claude Code**: wrote all source files, tests, prompts, and documentation
  in this repository; executed all test runs, evaluation runs, and packaging.
- **Codex**: reviewed evaluation outputs, identified semantic defects (notably
  the contradicted-verdict consistency bug), provided the final fix specification,
  and authorized each phase transition.
- **Human engineer**: problem framing, strategy priority weighting, final
  submission decision.

All code in `code/` was reviewed and accepted by both the human engineer and
Codex before the final commit.
