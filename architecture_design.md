# Architecture Design

Status: design only; no implementation is included.

## 1. Problem Understanding

### Inputs

The system processes one row from `dataset/claims.csv` at a time:

| Input | Meaning |
|---|---|
| `user_id` | Key used to retrieve the claimant's history |
| `image_paths` | One to three semicolon-separated paths relative to `dataset/` |
| `user_claim` | A short, sometimes multilingual or verbose conversation describing the claim |
| `claim_object` | Declared object class: `car`, `laptop`, or `package` |

Two reference datasets provide context:

- `user_history.csv` supplies historical counts, flags, and summaries by `user_id`.
- `evidence_requirements.csv` supplies global and object/issue-specific minimum visual evidence requirements.

Images are primary evidence. A claim may contain several images with different quality, distance, or relevance. Conversation text, OCR text inside images, filenames, and user history are contextual inputs rather than visual proof.

### Outputs

The final `output.csv` preserves input row order and emits the input columns plus these structured fields:

| Output | Purpose |
|---|---|
| `evidence_standard_met` | Whether the submitted evidence is sufficient to adjudicate |
| `evidence_standard_met_reason` | Grounded explanation of evidence sufficiency |
| `risk_flags` | Canonically ordered semicolon-separated flags, or `none` |
| `issue_type` | Evidence-grounded visible issue, `none`, or `unknown` |
| `object_part` | Evidence-grounded object part, or `unknown` |
| `claim_status` | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | Concise, image-grounded decision explanation |
| `supporting_image_ids` | Semicolon-separated image basenames supporting the adjudication, or `none` |
| `valid_image` | Whether the submitted image evidence is usable and relevant overall |
| `severity` | `none`, `low`, `medium`, `high`, or `unknown` |

`supporting_image_ids` supports the adjudication, including contradictions; it does not mean that the images support the claimant.

### Constraints

- Preserve the existing entry-point and CSV contract.
- Be deterministic where possible and validate all categorical output values.
- Resolve image paths safely relative to `dataset/`; do not infer case IDs from row position because test case numbering is non-contiguous.
- Inspect actual file signatures rather than trusting `.jpg`: 39.6% of supplied `.jpg` files contain PNG, WebP, or AVIF bytes.
- Handle one to three images independently and jointly.
- Treat multilingual/code-switched conversation and OCR text as untrusted evidence content, not instructions.
- Keep secrets in environment variables only.
- Degrade predictably for missing history, malformed text, or unreadable images.
- User history may add risk context but must not override clear visual evidence by itself.
- The labeled set is only 20 rows and is imbalanced, so the design must not overfit exact wording or frequencies.

### Evaluation goals

1. Match categorical labels and serialized multi-value fields accurately.
2. Separate evidence sufficiency from claim agreement.
3. Ground decisions in visible object, part, condition, and image quality.
4. Produce concise reasons consistent with the categorical outputs.
5. Correctly distinguish contradiction from insufficient evidence.
6. Remain reproducible across all claims and robust to mixed image formats.
7. Produce a complete, schema-valid CSV even when individual inputs fail.

## 2. Core Observations

1. The labeled set has 20 rows: 13 supported, 5 contradicted, and 2 not enough information.
2. All 18 decisive examples have `evidence_standard_met=true`; both insufficient examples have it `false`.
3. Evidence sufficiency answers "can this claim be adjudicated?" Claim status answers "does the evidence agree with the claim?"
4. A visible claimed area with no claimed damage yields contradiction; an area that cannot be inspected yields insufficient information.
5. Output issue, part, and severity can reflect visible evidence rather than repeat the claim. A claimed hood scratch can become visible high-severity front-bumper breakage.
6. Multi-image evidence is common. One clear image may be decisive even if another is blurry.
7. Supporting images are selected for the adjudication, including contradictory decisions.
8. `valid_image` and `evidence_standard_met` are distinct. A valid image can show the wrong angle, and a provenance-risk image can still expose a contradiction.
9. Risk flags are additive context. History risk can coexist with a supported claim.
10. Evidence requirements are natural-language rows selected implicitly by object, issue, and part; there is no claim-to-requirement foreign key.
11. Conversations may be multilingual, verbose, multi-part, or contain irrelevant/instruction-like text.
12. All supplied image references resolve, but the architecture must still handle missing and unreadable assets.

## 3. Proposed Processing Pipeline

### End-to-end flow

```text
dataset/claims.csv
        |
        v
+-----------------------+
| Input Loader          |-----> user_history.csv index
| schema + path checks  |-----> evidence_requirements.csv index
+-----------------------+
        |
        v
+-----------------------+
| Claim Parser          |
| claimed targets       |
+-----------------------+
        |
        +-----------------------------+
        |                             |
        v                             v
+-----------------------+    +-----------------------+
| Image Preflight       |    | Requirement Selector  |
| signature + decode    |    | global + applicable   |
+-----------------------+    +-----------------------+
        |
        v
+----------------------------------------------------+
| Joint Image Analyzer                                |
| per-image findings + cross-image synthesis          |
+----------------------------------------------------+
        |
        +-----------------------------+
        |                             |
        v                             v
+-----------------------+    +-----------------------+
| Evidence Validator    |    | Risk Assessor         |
| reviewability         |    | image + history risk  |
+-----------------------+    +-----------------------+
        |                             |
        +--------------+--------------+
                       v
              +-----------------------+
              | Decision Engine       |
              | reconcile + explain   |
              +-----------------------+
                       |
                       v
              +-----------------------+
              | Output Validator      |
              | enums + consistency   |
              +-----------------------+
                       |
                       v
              +-----------------------+
              | Ordered CSV Writer    |
              +-----------------------+
                       |
                       v
                    output.csv
```

### Per-claim control flow

```text
Claim Row
   |
   +--> Parse conversation into one or more claimed targets
   |
   +--> Resolve and decode every referenced image by file signature
   |
   +--> Select global and target-specific evidence requirements
   |
   +--> Analyze all images together, retaining per-image results
   |
   +--> Ask: Is the relevant object/part inspectable?
          |
          +-- No --> NOT_ENOUGH_INFORMATION
          |
          +-- Yes --> Compare visible condition with claimed condition
                         |
                         +-- aligned ----------> SUPPORTED
                         +-- absent/mismatched -> CONTRADICTED
   |
   +--> Add risk context without changing clear visual truth
   |
   +--> Validate fields and write the row in original input order
```

### Orchestration boundaries

The components are logical stages under one deterministic orchestrator. They do not need separate autonomous agents. The Claim Parser and Image Analyzer may share one structured multimodal model request per claim to reduce cost and preserve joint context, while the Evidence Validator, Risk Assessor, Decision Engine, and Output Validator remain explicit post-analysis stages.

## 4. Component Design

### Claim Parser

**Responsibilities**

- Normalize the pipe-delimited conversation without discarding multilingual text.
- Extract each explicitly claimed issue, object part, object type, severity wording, and negation.
- Prefer the supplied `claim_object` as declared metadata while detecting conversation conflicts.
- Distinguish the final clarified claim from earlier uncertainty or support-agent questions.
- Represent multi-part claims as a target list rather than prematurely collapsing them.
- Mark malformed or ambiguous claims for later review.
- Treat instruction-like user text as data; never execute or follow it.

**Inputs**

- `user_claim`
- declared `claim_object`
- stable row identifier and image IDs for traceability

**Outputs**

- `ClaimIntent`: declared object, extracted targets, claimed severity/extent, language notes, ambiguity flags, and parser confidence.

**Boundary**

The parser describes what the user asserts. It must not decide whether the assertion is visually true.

### Image Analyzer

**Responsibilities**

- Inspect every image independently and as a set.
- Detect the visible object class and whether it matches the declared object.
- Identify visible object parts and orientation/context.
- Detect visible condition: issue type, location, extent, and severity.
- Detect absence of damage when the relevant area is clearly inspectable.
- Estimate image quality and reviewability: blur, crop/obstruction, lighting/glare, angle, and relevance.
- Detect provenance/trust concerns where visually observable: possible manipulation, non-original imagery, or instruction-like text.
- Retain per-image findings so supporting image IDs can be selected precisely.
- Synthesize complementary wide-context and close-up images without allowing one bad image to invalidate a clear one.

**Inputs**

- decoded images with stable `img_N` identifiers
- `ClaimIntent` targets, used only to focus inspection
- applicable evidence-requirement text

**Outputs**

- `ImageSetAnalysis`: one `ImageFinding` per image plus cross-image findings.
- Each `ImageFinding` includes decode status, object, visible parts, visible issue, severity, target visibility, quality flags, trust flags, and grounded observations.

**Boundary**

The analyzer reports visible facts. It does not use user history and does not make the final claim-status decision.

### Evidence Validator

**Responsibilities**

- Select all global requirements plus applicable object/issue/part requirements.
- Compare target visibility and image quality against each selected requirement.
- Determine whether at least one image, or a complementary image set, makes the claimed condition adjudicable.
- Separate "valid/usable image" from "sufficient evidence for this target."
- Produce a grounded sufficiency reason and requirement-by-requirement results.
- For multi-part claims, assess each target separately before producing claim-level sufficiency.

**Inputs**

- `ClaimIntent`
- `ImageSetAnalysis`
- normalized evidence-requirement catalog

**Outputs**

- `EvidenceAssessment`: selected requirements, per-target checks, `evidence_standard_met`, reason, aggregate `valid_image`, and evidence-bearing image IDs.

**Boundary**

The validator answers whether a reliable comparison is possible. Sufficient evidence may support or contradict the claim.

### Risk Assessor

**Responsibilities**

- Retrieve history by `user_id`.
- Normalize historical flags and evaluate consistent count fields.
- Combine history context with image-quality, relevance, mismatch, provenance, and embedded-text concerns from upstream stages.
- Emit only allowed atomic risk flags in canonical order.
- Add `manual_review_required` when explicit history/source rules or unresolved trust concerns justify it.
- Keep history risk advisory: never reverse clear visual evidence solely because a user has prior claims.

**Inputs**

- `user_id`
- optional history record
- `ClaimIntent`
- `ImageSetAnalysis`
- `EvidenceAssessment`

**Outputs**

- `RiskAssessment`: ordered atomic flags, history availability, short context summary, and manual-review rationale.

**Boundary**

The risk assessor adds context. It cannot mark a claim contradicted merely because the user is high risk.

### Decision Engine

**Responsibilities**

- Reconcile claimed targets with visible, inspectable findings.
- Produce `claim_status`, `claim_status_justification`, and `supporting_image_ids`.
- Finalize evidence-grounded `issue_type`, `object_part`, and `severity` for the output contract.
- Preserve the distinction among visible alignment, visible contradiction, and uninspectable evidence.
- Select only images that materially support the adjudication.
- Incorporate risk context after the visual conclusion, without letting it override visual truth.
- Resolve multi-target claims consistently and flag mixed or ambiguous target outcomes for manual review.

**Inputs**

- original row and `ClaimIntent`
- `ImageSetAnalysis`
- `EvidenceAssessment`
- `RiskAssessment`

**Outputs**

- `DecisionRecord` containing every derived output field and trace references to supporting findings.

**Boundary**

The engine performs reconciliation and output selection, not raw image perception or history extraction.

### Supporting components

**Input Loader and Reference Indexes**

- Parse CSVs with a real CSV parser, preserve row order, split only documented semicolon-list fields, and index history/requirements once.

**Image Preflight**

- Resolve paths, reject traversal outside `dataset/`, inspect magic bytes, decode JPEG/PNG/WebP/AVIF, normalize orientation/color, and retain original IDs.

**Output Validator**

- Enforce allowed enums, booleans, canonical flag ordering, valid supporting IDs, non-empty reasons, and cross-field consistency before serialization.

**Ordered CSV Writer**

- Write exactly one row per input row in original order, with stable quoting and semicolon serialization.

## 5. Data Flow

### Claim row to parsed intent

```json
{
  "input_row": {
    "row_index": 0,
    "user_id": "user_002",
    "image_paths": [
      "images/test/case_001/img_1.jpg",
      "images/test/case_001/img_2.jpg",
      "images/test/case_001/img_3.jpg"
    ],
    "user_claim": "...front bumper...left headlight...",
    "claim_object": "car"
  },
  "claim_intent": {
    "declared_object": "car",
    "targets": [
      {"part": "front_bumper", "issue": "damage", "claimed_severity": "unknown"},
      {"part": "headlight", "issue": "damage", "claimed_severity": "unknown"}
    ],
    "ambiguities": [],
    "untrusted_instruction_detected": false
  }
}
```

### Image analysis contract

```json
{
  "images": [
    {
      "image_id": "img_1",
      "decode_status": "ok",
      "native_format": "avif",
      "object": "car",
      "visible_parts": ["front_bumper", "headlight"],
      "target_visibility": {
        "front_bumper": "clear",
        "headlight": "clear"
      },
      "visible_issue": {
        "type": "broken_part",
        "part": "front_bumper",
        "severity": "high"
      },
      "quality_flags": [],
      "trust_flags": [],
      "observation": "Front bumper is visibly broken; headlight condition is separately inspectable."
    }
  ],
  "set_summary": {
    "declared_object_match": true,
    "complementary_views": true,
    "instruction_text_is_evidence_only": true
  }
}
```

The values above illustrate the contract, not a prediction for that dataset row.

### Evidence assessment contract

```json
{
  "selected_requirements": [
    "REQ_GENERAL_OBJECT_PART",
    "REQ_GENERAL_MULTI_IMAGE",
    "REQ_CAR_GLASS_LIGHT_MIRROR",
    "REQ_REVIEW_TRUST"
  ],
  "target_checks": [
    {
      "part": "headlight",
      "inspectable": true,
      "evidence_image_ids": ["img_1", "img_2"]
    }
  ],
  "evidence_standard_met": true,
  "evidence_standard_met_reason": "The relevant car parts are visible clearly enough across the submitted views to evaluate the claim.",
  "valid_image": true
}
```

### Risk assessment contract

```json
{
  "history_found": true,
  "history_flags": ["none"],
  "image_flags": ["blurry_image"],
  "decision_context_flags": [],
  "risk_flags": ["blurry_image"],
  "manual_review_reason": null
}
```

### Final decision contract

```json
{
  "evidence_standard_met": true,
  "evidence_standard_met_reason": "The claimed part is clearly visible and can be evaluated.",
  "risk_flags": ["none"],
  "issue_type": "dent",
  "object_part": "rear_bumper",
  "claim_status": "supported",
  "claim_status_justification": "The image clearly shows a dent on the claimed rear bumper.",
  "supporting_image_ids": ["img_1"],
  "valid_image": true,
  "severity": "medium"
}
```

Before CSV output, arrays become canonical semicolon-delimited strings and booleans become lowercase `true`/`false`.

## 6. Decision Logic

The following is high-level decision behavior, not executable prediction logic.

### Supported

A claim is supported when:

- the relevant object and claimed part are sufficiently inspectable;
- the visible issue aligns materially with the claimed issue and location; and
- at least one submitted image provides grounded evidence for the conclusion.

Minor image-quality or history risks may remain as flags if another image supplies clear evidence.

### Contradicted

A claim is contradicted when the evidence is sufficiently inspectable but shows a material conflict, such as:

- no claimed damage on a clearly visible target part;
- a different issue, object part, or object;
- materially different severity/extent; or
- evidence that directly establishes the claimed condition is not present despite adequate visibility.

Contradictory decisions still list images that establish the conflict.

### Not enough information

A claim has insufficient information when the evidence cannot reliably answer the claim, such as:

- the relevant part is outside the frame or shown from the wrong angle;
- crop, obstruction, blur, glare, or darkness prevents inspection;
- contents required for a missing/damaged-item claim are not visible;
- all referenced images are missing or unreadable; or
- the conversation is too malformed to identify a review target and visual evidence cannot resolve it.

The architectural invariant is:

```text
Can the claimed condition be adjudicated from the submitted evidence?
    No  -> not_enough_information
    Yes -> Does visible evidence materially align with the claim?
              Yes -> supported
              No  -> contradicted
```

### Multi-target claims

Targets are assessed independently. A claim-level result is supported only when all material claimed targets are supported; any decisive material contradiction prevents full support. If one material target is adjudicable and another is not, the record should carry `manual_review_required`, preserve per-target trace data, and use the contract's conservative claim-level outcome. Because the labeled set does not establish the exact aggregation convention, this policy must be validated during evaluation rather than assumed from one example.

## 7. Error Handling

| Failure | Handling | Output effect |
|---|---|---|
| Unreadable/corrupt image | Record per-image decode failure; continue with remaining images | If another relevant image is sufficient, decide from it and flag review; otherwise insufficient with `valid_image=false` |
| Unsupported native image format | Detect by signature, attempt supported normalization, never trust extension | Same as unreadable if normalization fails; preserve diagnostic internally |
| Missing image path | Record missing asset without aborting the batch | Use remaining images if sufficient; otherwise insufficient and manual review |
| Path outside dataset root | Reject as unsafe | Treat as missing/unusable evidence and flag manual review |
| Malformed `image_paths` cell | Parse conservatively, reject empty/invalid entries, retain valid ones | Decide only if retained images are sufficient |
| Malformed conversation | Preserve raw text, use declared object, attempt target extraction, record ambiguity | If image evidence cannot identify the target, insufficient and manual review |
| Missing history row | Substitute explicit `history_found=false`; do not infer risk | Do not penalize claim status; add manual review only if policy requires missing context |
| Malformed history counts | Ignore inconsistent derived conclusions and retain raw record diagnostically | Do not override visual evidence; optionally manual review |
| Missing requirements | Apply global visibility/reviewability requirements and record catalog gap | Conservative sufficiency decision; manual review for uncovered target |
| Model timeout/rate limit | Retry with bounded exponential backoff; use cache before retry | After exhausted retries, emit a schema-valid insufficient/manual-review row rather than drop it |
| Invalid model structure/enums | Validate and request bounded structured repair; never accept free-form schema drift | Fallback to conservative schema-valid output after repair limit |
| Partial batch failure | Isolate by row and checkpoint successful results | Continue other rows; restore original row order at write time |

Errors and diagnostics belong in runtime logs or cache metadata, not in user-facing justifications unless they directly explain insufficient evidence.

## 8. Cost and Latency Considerations

### Expected workload

For the supplied test data:

- 44 claim rows
- 82 images
- 13 one-image, 24 two-image, and 7 three-image claims

### Recommended call pattern

| Stage | Recommended calls |
|---|---:|
| CSV/reference loading | 1 local pass per file |
| Image signature/decode/normalization | 82 local operations |
| Joint multimodal analysis | Up to 44 calls, one per claim |
| Evidence/risk/decision validation | Local deterministic stages where possible |
| Repair/retry | Only for invalid/failed responses |

The preferred baseline is one multimodal request per claim containing its conversation, selected requirements, and all one-to-three images. This preserves cross-image context and avoids 82 independent model calls plus a second synthesis call.

### Batching opportunities

- Preflight and normalize images concurrently with a bounded worker pool.
- Process independent claims concurrently, bounded by provider rate and token limits.
- Batch text-only history lookup and requirement selection locally.
- Avoid combining unrelated claims into one multimodal request; cross-claim batching complicates traceability and failure isolation.

### Caching opportunities

- Cache image normalization by content hash, native format, and transform version.
- Cache multimodal analysis by hashes of image bytes, normalized claim intent, requirement IDs, model/version, and analysis-contract version.
- Cache parsed claim intent by normalized conversation hash and parser version.
- Load history and requirement indexes once per run.
- Persist successful per-row results so reruns only process changed or failed rows.

### Cost/quality controls

- Resize only above a conservative maximum while retaining enough detail for small cracks, scratches, seals, and labels.
- Send thumbnails plus higher-resolution relevant views only if the provider supports multi-resolution input.
- Keep structured outputs concise; avoid returning long descriptions not used by the contract.
- Use bounded retries and schema repair rather than unbounded agent loops.
- Track model and prompt/config versions in cache metadata for reproducibility.

Latency is dominated by the 44 multimodal calls. With safe concurrency, wall-clock time approaches several call-latency waves rather than 44 serial calls, while maintaining per-claim isolation.

## 9. Interview Defense

### Claim Parser

**Why it exists:** The model must know the exact asserted target before comparing images. Conversations contain clarification, negation, multiple issues, and multiple languages.

**Why separated:** Claim intent is an assertion, while image analysis is observation. Keeping them distinct makes mismatches explainable.

**Alternatives considered:** Pure keyword/regex extraction is cheap and deterministic but brittle for code-switching, long conversations, and negation. Asking the vision stage to infer the claim without a typed intent obscures failures.

**Tradeoff:** A dedicated logical stage adds a contract and confidence handling, but substantially improves traceability. It can share the same physical model call to avoid extra cost.

### Image Analyzer

**Why it exists:** Images are the primary source of truth and must be evaluated per image and jointly.

**Why separated:** Visual perception should not be contaminated by user-history priors or final-status expectations.

**Alternatives considered:** Traditional CV detectors are fast and reproducible but do not cover the broad part/damage taxonomy without training data. Separate calls per image improve isolation but increase cost and lose cross-view context.

**Tradeoff:** A joint VLM analysis is flexible and context-aware but less deterministic. Typed outputs, per-image findings, low randomness, and validation reduce that risk.

### Evidence Validator

**Why it exists:** The central dataset distinction is between decisive evidence and evidence that cannot answer the claim.

**Why separated:** Evidence sufficiency is independent of whether the claim is true. Combining it with status invites the common error "bad claim equals insufficient evidence."

**Alternatives considered:** Letting the VLM output status directly is simpler but makes contradiction versus insufficiency inconsistent and leaves evidence requirements unenforced.

**Tradeoff:** Explicit requirement checks add mapping complexity because requirements are natural language, but they make decisions auditable and data-driven.

### Risk Assessor

**Why it exists:** The contract requires image, provenance, and user-history risk flags.

**Why separated:** History must add context without biasing visual perception or overriding clear evidence.

**Alternatives considered:** Feeding all history into the initial image analysis is simpler but risks anchoring the model on claimant reputation. Ignoring history loses required flags.

**Tradeoff:** Late fusion preserves evidence primacy but requires explicit conflict handling and canonical flag ordering.

### Decision Engine

**Why it exists:** Parsed assertions, visible facts, evidence sufficiency, and risk context must be reconciled into a coherent output row.

**Why separated:** Central reconciliation keeps status and reasons consistent and prevents each component from inventing its own outcome.

**Alternatives considered:** End-to-end free-form model prediction minimizes components but is harder to validate, cache, debug, and defend. Fully handwritten rules are deterministic but too brittle for nuanced visual mismatch.

**Tradeoff:** A structured reconciliation stage adds design complexity but offers the best balance of flexibility and reproducibility.

### Image Preflight and Output Validator

**Why they exist:** Mixed native formats and strict CSV enums are concrete failure modes independent of model intelligence.

**Why separated:** Format decoding and schema validation are deterministic infrastructure concerns.

**Alternatives considered:** Trusting filenames/provider coercion is simpler but fails for the 44 mislabeled files. Trusting raw model output risks invalid rows.

**Tradeoff:** Local normalization adds CPU/storage overhead but prevents avoidable provider and evaluation failures.

### Deterministic Orchestrator

**Why it exists:** It controls row order, retries, caching, concurrency, validation, and failure isolation.

**Why separated:** Operational control should not be delegated to a probabilistic model.

**Alternatives considered:** Autonomous agents could plan and debate each claim, but the task has a fixed schema and short workflow. A single monolithic script is simpler initially but mixes concerns and weakens testability.

**Tradeoff:** A staged orchestrator has more explicit contracts but lower variance, bounded cost, and clearer diagnostics.

## 10. Final Recommendation

Use a **multi-stage pipeline under one deterministic orchestrator**, with a single joint multimodal analysis call per claim where practical.

Do not use a multi-agent architecture. The workload is a fixed, schema-driven adjudication process, not an open-ended planning problem. Multiple agents would add latency, cost, coordination failure, and nondeterminism without a corresponding evidence-quality benefit.

Do not use a fully monolithic single-agent decision either. The labeled examples show that claim parsing, visual findings, evidence sufficiency, history risk, and final status represent different concepts. Collapsing them makes contradiction versus insufficient evidence harder to control and defend.

The recommended balance is:

```text
Deterministic orchestration
    + typed claim intent
    + joint per-claim multimodal findings
    + explicit requirement validation
    + late-fused risk context
    + constrained decision reconciliation
    + strict output validation
```

This design is cost-bounded (approximately one multimodal call per claim), supports caching and concurrency, handles the mixed image formats, preserves visual evidence as the source of truth, and exposes enough intermediate structure to evaluate and debug every output field.
