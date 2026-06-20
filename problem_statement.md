# Multi-Modal Evidence Review

Build a system that verifies damage claims using images, a short claim conversation, user history, and minimum evidence requirements.

Each claim is about one of three object types:

- `car`
- `laptop`
- `package`

Your system must decide whether the submitted images support the user's claim, contradict it, or do not provide enough information.

The images are the primary source of truth. The user conversation defines what needs to be checked. User history can add risk context, but should not override clear visual evidence by itself.

## What the system should do

For each claim, your system should:

- extract the actual damage claim from the conversation
- inspect one or more submitted images
- decide whether the image evidence is sufficient
- identify the visible issue type
- identify the relevant object part
- decide whether the claim is supported, contradicted, or lacks enough information
- select the image IDs that support the decision
- flag image quality, mismatch, authenticity, or user-history risks
- estimate severity
- produce short justifications grounded in the images

## Files provided

You will receive:

1. `dataset/sample_claims.csv`  
   Labeled examples with inputs and expected outputs. Use this to understand the expected behavior and evaluate your system.

2. `dataset/claims.csv`  
   Input-only rows. Run your system on this file and produce `output.csv`.

3. `dataset/user_history.csv`  
   Historical claim counts and risk patterns for each user.

4. `dataset/evidence_requirements.csv`  
   A minimum image evidence checklist by object and issue family.

5. `dataset/images/sample/` and `dataset/images/test/`  
   Image folders referenced by the CSV files.

Multiple images in `image_paths` are separated by semicolons:

```text
images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg
```

The image ID is the filename without extension, such as `img_1`.

## Input schema

Each row in `claims.csv` represents one damage claim.

Input fields:

- `user_id`: user submitting the claim; use this to look up `user_history.csv`
- `image_paths`: one or more submitted image paths
- `user_claim`: chat transcript about the issue
- `claim_object`: `car`, `laptop`, or `package`

## Evidence requirements schema

`dataset/evidence_requirements.csv` contains:

- `requirement_id`: identifier for the rule
- `claim_object`: `car`, `laptop`, `package`, or `all`
- `applies_to`: issue family, such as `dent or scratch`
- `minimum_image_evidence`: minimum visual evidence needed to evaluate that kind of claim

## User history schema

`dataset/user_history.csv` contains:

- `user_id`
- `past_claim_count`
- `accept_claim`
- `manual_review_claim`
- `rejected_claim`
- `last_90_days_claim_count`
- `history_flags`
- `history_summary`

Use history to add risk context through `risk_flags` and justifications.

## Required output

For each row in `claims.csv`, generate one row in `output.csv`.

Required columns, in order:

- `user_id`
- `image_paths`
- `user_claim`
- `claim_object`
- `evidence_standard_met`
- `evidence_standard_met_reason`
- `risk_flags`
- `issue_type`
- `object_part`
- `claim_status`
- `claim_status_justification`
- `supporting_image_ids`
- `valid_image`
- `severity`

## Output meaning

- `evidence_standard_met`: `true` if the image set is sufficient to evaluate the claim; otherwise `false`
- `evidence_standard_met_reason`: short reason for the evidence decision
- `risk_flags`: semicolon-separated risk flags, or `none`
- `issue_type`: visible issue type
- `object_part`: relevant object part
- `claim_status`: final decision: `supported`, `contradicted`, or `not_enough_information`
- `claim_status_justification`: concise image-grounded explanation; mention relevant image IDs when helpful
- `supporting_image_ids`: image IDs supporting the decision, separated by semicolons; use `none` if no image is sufficient
- `valid_image`: `true` if the image set is usable for automated review; otherwise `false`
- `severity`: `none`, `low`, `medium`, `high`, or `unknown`

## Allowed values

Use the closest matching value from these lists.

`claim_status`: `supported`, `contradicted`, `not_enough_information`

`issue_type`: `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`, `missing_part`, `torn_packaging`, `crushed_packaging`, `water_damage`, `stain`, `none`, `unknown`

Car `object_part`: `front_bumper`, `rear_bumper`, `door`, `hood`, `windshield`, `side_mirror`, `headlight`, `taillight`, `fender`, `quarter_panel`, `body`, `unknown`

Laptop `object_part`: `screen`, `keyboard`, `trackpad`, `hinge`, `lid`, `corner`, `port`, `base`, `body`, `unknown`

Package `object_part`: `box`, `package_corner`, `package_side`, `seal`, `label`, `contents`, `item`, `unknown`

`risk_flags`: `none`, `blurry_image`, `cropped_or_obstructed`, `low_light_or_glare`, `wrong_angle`, `wrong_object`, `wrong_object_part`, `damage_not_visible`, `claim_mismatch`, `possible_manipulation`, `non_original_image`, `text_instruction_present`, `user_history_risk`, `manual_review_required`

Use `issue_type=none` when the relevant part is visible and no issue is present. Use `unknown` when the issue or part cannot be determined.

## Evaluation requirement

Your `code.zip` must include an `evaluation/` folder.

Use `dataset/sample_claims.csv` to evaluate your system before producing final predictions for `dataset/claims.csv`.

## Operational analysis

Include a short operational analysis in `evaluation/evaluation_report.md`.

Report:

- approximate number of model calls for sample and test processing
- approximate input/output token usage
- number of images processed
- approximate cost to process the full test set, with pricing assumptions
- approximate latency or runtime
- TPM/RPM considerations and any batching, throttling, caching, or retry strategy

You are not expected to optimize perfectly, but your solution should show that you considered cost, latency, rate limits, and unnecessary repeated calls.

## Requirements:
- Must read the provided CSV files and local images.
- Must produce output.csv with the exact schema in problem_statement.md.
- Must include an evaluation workflow
- Must avoid hardcoded test labels or file-specific answers.
- Beyond that you are free to bring your own approach and innovation: VLMs, LLMs, structured prompting, evaluation pipelines, model comparison, or anything else.

## Submission

Submit:

| File | Description |
|---|---|
| `code.zip` | Full runnable solution, prompts/configs, README, and `evaluation/` folder. |
| `output.csv` | Predictions for all rows in `dataset/claims.csv`. |
| `chat_transcript` | Conversation transcript showing how you developed or used the system. |

These are the must-haves. Beyond that, participants are encouraged to improve retrieval, prompting, evaluation, confidence handling, batching, caching, or review logic.
