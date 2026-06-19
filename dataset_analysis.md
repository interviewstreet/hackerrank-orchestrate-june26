# Dataset Analysis

Analysis timestamp: 2026-06-19 (Asia/Calcutta)

## Scope

This document describes the repository and supplied dataset only. It does not propose an architecture, prediction logic, prompts, or implementation.

## Repository Structure

```text
.
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
|-- problem_statement.md
|-- code/
|   |-- main.py
|   `-- evaluation/
|       `-- main.py
`-- dataset/
    |-- claims.csv
    |-- evidence_requirements.csv
    |-- output.csv
    |-- sample_claims.csv
    |-- user_history.csv
    `-- images/
        |-- sample/
        |   `-- case_NNN/img_N.jpg
        `-- test/
            `-- case_NNN/img_N.jpg
```

The repository contains starter entry points under `code/`, task documentation, four input/reference CSVs, an existing `dataset/output.csv`, and image assets. This analysis did not modify or assess the starter code or output predictions.

## CSV Overview

| File | Data rows (header excluded) | Columns | Role |
|---|---:|---:|---|
| `dataset/sample_claims.csv` | 20 | 14 | Labeled development examples |
| `dataset/claims.csv` | 44 | 4 | Unlabeled claims to process |
| `dataset/user_history.csv` | 47 | 8 | User-level historical context |
| `dataset/evidence_requirements.csv` | 11 | 4 | Evidence guidance by object/issue family |

All four CSVs have headers and no missing/blank field values.

## `sample_claims.csv`

### Columns and inferred types

CSV stores every value as text; the semantic types below are inferred from content.

| Column | Inferred type | Description |
|---|---|---|
| `user_id` | string identifier | Join key to user history |
| `image_paths` | semicolon-delimited string list | Paths relative to `dataset/` |
| `user_claim` | string | Pipe-delimited conversation text |
| `claim_object` | categorical string | `car`, `laptop`, or `package` |
| `evidence_standard_met` | boolean encoded as `true`/`false` | Labeled evidence sufficiency |
| `evidence_standard_met_reason` | string | Free-text rationale |
| `risk_flags` | semicolon-delimited categorical list | One or more risk labels, or `none` |
| `issue_type` | categorical string | Labeled issue family |
| `object_part` | categorical string | Labeled object component |
| `claim_status` | categorical string | Outcome label |
| `claim_status_justification` | string | Free-text outcome rationale |
| `supporting_image_ids` | semicolon-delimited string list | Basenames without extensions |
| `valid_image` | boolean encoded as `true`/`false` | Image validity label |
| `severity` | categorical string | Damage severity label |

### First 3 rows

```json
[
  {
    "user_id": "user_001",
    "image_paths": "images/sample/case_001/img_1.jpg",
    "user_claim": "Customer: Hi, I found new damage on my car after it was parked outside overnight. | Support: Sorry to hear that. Can you describe what changed? | Customer: The back of the car has a dent now. It was not there before. | Support: Did anything else break or is it mostly body damage? | Customer: Mostly the rear bumper area. I attached the photo I took this morning.",
    "claim_object": "car",
    "evidence_standard_met": "true",
    "evidence_standard_met_reason": "The rear bumper is visible and the dent can be verified from the submitted image.",
    "risk_flags": "none",
    "issue_type": "dent",
    "object_part": "rear_bumper",
    "claim_status": "supported",
    "claim_status_justification": "The image clearly shows a dent on the rear bumper and the user history does not add risk.",
    "supporting_image_ids": "img_1",
    "valid_image": "true",
    "severity": "medium"
  },
  {
    "user_id": "user_002",
    "image_paths": "images/sample/case_002/img_1.jpg;images/sample/case_002/img_2.jpg",
    "user_claim": "Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type ka damage report karna chahte hain? | Customer: Front side par mark aa gaya hai, bumper ke upar. | Support: Light damage hai ya body par scratch? | Customer: Light theek hai, front bumper par scratch hai. Photos upload kar diye hain.",
    "claim_object": "car",
    "evidence_standard_met": "true",
    "evidence_standard_met_reason": "The full front view provides context and the close-up image shows the scratch on the front bumper.",
    "risk_flags": "none",
    "issue_type": "scratch",
    "object_part": "front_bumper",
    "claim_status": "supported",
    "claim_status_justification": "The close-up image shows a visible scratch on the claimed front bumper.",
    "supporting_image_ids": "img_1",
    "valid_image": "true",
    "severity": "low"
  },
  {
    "user_id": "user_004",
    "image_paths": "images/sample/case_003/img_1.jpg;images/sample/case_003/img_2.jpg",
    "user_claim": "Customer: I am opening a claim for my windshield. | Support: What happened? | Customer: A small stone hit it while I was driving and now there is a crack spreading from that spot. | Support: Is the car otherwise okay? | Customer: Yes, this is only about the front glass. I added the pictures I have.",
    "claim_object": "car",
    "evidence_standard_met": "true",
    "evidence_standard_met_reason": "The windshield is visible and the close-up image shows clear crack lines.",
    "risk_flags": "none",
    "issue_type": "crack",
    "object_part": "windshield",
    "claim_status": "supported",
    "claim_status_justification": "The image set supports the claim because the windshield crack is visible in the close-up.",
    "supporting_image_ids": "img_1",
    "valid_image": "true",
    "severity": "medium"
  }
]
```

### Missing values

Every column has `0` missing values across 20 rows.

### Useful distributions

| Field | Distribution |
|---|---|
| `claim_object` | car 8; laptop 6; package 6 |
| `claim_status` | supported 13; contradicted 5; not_enough_information 2 |
| `evidence_standard_met` | true 18; false 2 |
| `valid_image` | true 18; false 2 |
| `severity` | medium 11; low 4; none 2; unknown 2; high 1 |
| images per row | 1 image: 11 rows; 2 images: 9 rows |

`issue_type`: broken_part 3, crack 3, dent 3, unknown 3, none 2, scratch 2, crushed_packaging 1, stain 1, torn_packaging 1, water_damage 1.

Atomic `risk_flags` counts (multi-flag rows contribute to each flag): none 11, manual_review_required 7, user_history_risk 6, damage_not_visible 4, claim_mismatch 3, and one each of blurry_image, cropped_or_obstructed, non_original_image, text_instruction_present, wrong_angle, and wrong_object.

There are 16 distinct `object_part` values. The most frequent are `front_bumper`, `rear_bumper`, `screen`, and `seal` with 2 rows each; all other values occur once.

## `claims.csv`

### Columns and inferred types

| Column | Inferred type | Description |
|---|---|---|
| `user_id` | string identifier | Join key to user history |
| `image_paths` | semicolon-delimited string list | One to three paths relative to `dataset/` |
| `user_claim` | string | Pipe-delimited conversation text |
| `claim_object` | categorical string | `car`, `laptop`, or `package` |

### First 3 rows

```json
[
  {
    "user_id": "user_002",
    "image_paths": "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg;images/test/case_001/img_3.jpg",
    "user_claim": "Customer: Morning. I parked near office and later noticed something off in the front. | Agent: Is this about one part or multiple parts? | Customer: Two things, I think. The front bumper looks damaged and the left headlight also looks affected. | Agent: Should we review both as part of this claim? | Customer: Yes, front bumper and left headlight together.",
    "claim_object": "car"
  },
  {
    "user_id": "user_005",
    "image_paths": "images/test/case_003/img_1.jpg",
    "user_claim": "Customer: Need to file a car damage claim. | Agent: What part of the car? | Customer: Door. | Agent: Scratch, dent, or paint issue? | Customer: A deep dent on the door panel. It was not there before.",
    "claim_object": "car"
  },
  {
    "user_id": "user_004",
    "image_paths": "images/test/case_004/img_1.jpg;images/test/case_004/img_2.jpg",
    "user_claim": "Customer: A stone hit the front glass while driving. | Support: Are you reporting the windshield? | Customer: Yes. It looks shattered from my side. | Support: Any other part involved? | Customer: No, only the windshield shatter claim.",
    "claim_object": "car"
  }
]
```

### Missing values and distributions

Every column has `0` missing values across 44 rows.

| Field | Distribution |
|---|---|
| `claim_object` | car 18; laptop 13; package 13 |
| images per row | 1 image: 13 rows; 2 images: 24 rows; 3 images: 7 rows |

All 44 `image_paths` values and all 44 conversations are unique. Claims include multilingual or code-switched text (including English, Hindi/Hinglish, and Spanish), verbose conversations, multiple-part claims, and at least one instruction embedded inside claim text.

## `user_history.csv`

### Columns and inferred types

| Column | Inferred type | Description |
|---|---|---|
| `user_id` | string identifier | Unique user key |
| `past_claim_count` | non-negative integer | Total historical claims |
| `accept_claim` | non-negative integer | Accepted historical claims |
| `manual_review_claim` | non-negative integer | Historically manually reviewed claims |
| `rejected_claim` | non-negative integer | Rejected historical claims |
| `last_90_days_claim_count` | non-negative integer | Recent claim count |
| `history_flags` | semicolon-delimited categorical list | User-level risk/review flags, or `none` |
| `history_summary` | string | Free-text history description |

### First 3 rows

| user_id | past_claim_count | accept_claim | manual_review_claim | rejected_claim | last_90_days_claim_count | history_flags | history_summary |
|---|---:|---:|---:|---:|---:|---|---|
| user_001 | 2 | 2 | 0 | 0 | 1 | none | Low-risk user with prior accepted car damage claims |
| user_002 | 4 | 3 | 1 | 0 | 2 | none | Mostly accepted vehicle claims with one manual review |
| user_003 | 1 | 1 | 0 | 0 | 0 | none | Limited history and no notable risk |

### Missing values and distributions

Every column has `0` missing values across 47 rows. All 47 `user_id` values are unique.

| Numeric field | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| `past_claim_count` | 0 | 14 | 3.96 |
| `accept_claim` | 0 | 4 | 1.96 |
| `manual_review_claim` | 0 | 4 | 1.02 |
| `rejected_claim` | 0 | 7 | 0.98 |
| `last_90_days_claim_count` | 0 | 9 | 2.11 |

Atomic `history_flags` counts are: none 22, user_history_risk 22, manual_review_required 11. Multi-flag rows contribute to each atomic count. There are 45 distinct history summaries; `New user with no prior claim history` occurs 3 times and every other summary is unique.

For all 47 users, `past_claim_count` equals `accept_claim + manual_review_claim + rejected_claim`.

## `evidence_requirements.csv`

### Columns and inferred types

| Column | Inferred type | Description |
|---|---|---|
| `requirement_id` | string identifier | Unique requirement key |
| `claim_object` | categorical string | `all`, `car`, `laptop`, or `package` |
| `applies_to` | categorical/free-text string | Issue or review family |
| `minimum_image_evidence` | string | Natural-language evidence requirement |

### First 3 rows

| requirement_id | claim_object | applies_to | minimum_image_evidence |
|---|---|---|---|
| REQ_GENERAL_OBJECT_PART | all | general claim review | The claimed object and relevant part should be visible clearly enough to inspect the claimed condition. |
| REQ_GENERAL_MULTI_IMAGE | all | multi-image rows | Each submitted image should be considered separately; at least one relevant image should show the claimed object or part clearly enough to evaluate the claim. |
| REQ_CAR_BODY_PANEL | car | dent or scratch | The claimed car panel or bumper should be visible from an angle where surface marks or deformation can be assessed. |

### Missing values and distributions

Every column has `0` missing values across 11 rows. All requirement IDs, `applies_to` values, and evidence descriptions are unique.

| `claim_object` | Requirements |
|---|---:|
| all | 3 |
| car | 3 |
| package | 3 |
| laptop | 2 |

The 11 `applies_to` families cover general review, multi-image rows, reviewability, car body damage/components/identity, laptop surface/body parts, and package exterior/labels/contents.

## Image Organization

```text
dataset/images/
|-- sample/
|   |-- case_001/
|   |   `-- img_1.jpg
|   |-- case_002/
|   |   |-- img_1.jpg
|   |   `-- img_2.jpg
|   `-- ... case_020/
`-- test/
    |-- case_001/
    |   |-- img_1.jpg
    |   |-- img_2.jpg
    |   `-- img_3.jpg
    `-- ... non-contiguous case numbers through case_056/
```

| Split | Case folders | Image files | Images per claim | Hidden metadata files |
|---|---:|---:|---|---:|
| sample | 20 | 29 | 11x one image; 9x two images | 2 `.DS_Store` |
| test | 44 | 82 | 13x one image; 24x two images; 7x three images | 3 `.DS_Store` |
| total | 64 | 111 | 1-3 images per claim | 5 `.DS_Store` |

### Naming conventions

- CSV paths use forward slashes and are relative to `dataset/`, for example `images/test/case_001/img_1.jpg`.
- Multiple paths are separated by semicolons without spaces.
- Case folders match `case_NNN` with zero-padded three-digit numbers.
- Image names match `img_N.jpg`, with numbering local to each case and starting at 1.
- Sample cases are contiguous from `case_001` through `case_020`.
- Test case numbering is not contiguous. The 44 present case folders span `case_001` through `case_056`; absent numbers are 002, 009, 012, 013, 015, 016, 021, 022, 023, 024, 033, and 035.
- All 111 referenced image paths exist. No JPEG-named image is unreferenced, and no case folder lacks a corresponding CSV row.

### File signatures

The `.jpg` extension is not a reliable indicator of the actual encoded format.

| Split | JPEG signature | PNG signature | WebP signature | AVIF signature |
|---|---:|---:|---:|---:|
| sample (29) | 18 | 5 | 6 | 0 |
| test (82) | 49 | 14 | 11 | 8 |
| total (111) | 67 | 19 | 17 | 8 |

Thus, 44 of 111 files (39.6%) are not JPEG-encoded despite having a `.jpg` filename. Five `.DS_Store` files are also present and are not referenced by the CSVs.

## Relationships Between Files

1. Each row in `sample_claims.csv` maps positionally to one `images/sample/case_NNN/` folder; each row in `claims.csv` maps through `image_paths` to one `images/test/case_NNN/` folder.
2. `image_paths` is the explicit claim-to-image relationship. A row may reference one, two, or three images.
3. `user_id` joins both claims files to `user_history.csv`. Every sample and test claim user has a matching history row.
4. Fourteen users occur in both the sample and test claim sets.
5. Five history users are unused by either claims file: `user_013`, `user_021`, `user_023`, `user_024`, and `user_035`.
6. `claim_object` connects a claim to object-specific rows in `evidence_requirements.csv`; the `all` rows are cross-object requirements.
7. Labeled output fields exist only in `sample_claims.csv`; `claims.csv` contains only the four input columns.
8. `supporting_image_ids` in the sample data refers to image basenames such as `img_1`, while `image_paths` contains full relative paths and extensions.

## Key Observations

1. The labeled set is small (20 rows) relative to the 44-row test set.
2. Both claim sets cover the same three object classes, with cars the largest class.
3. The sample outcomes are imbalanced: 13 of 20 are `supported`, compared with 5 `contradicted` and 2 `not_enough_information`.
4. Sample evidence/validity booleans are also imbalanced: 18 true and 2 false for each field.
5. Multi-image evidence is common: 9 of 20 sample rows and 31 of 44 test rows contain multiple images.
6. All claim users have history records, and the historical count components are internally consistent.
7. Risk flags and image/support IDs are multi-valued fields encoded inside single CSV cells with semicolons.
8. Conversations are pipe-delimited, multilingual/code-switched, and sometimes verbose or multi-issue.
9. Every referenced image exists and every image is referenced, but test case IDs are intentionally non-contiguous.
10. Filename extensions are misleading: 39.6% of `.jpg` paths contain PNG, WebP, or AVIF bytes.

## Potential Challenges

- Small and imbalanced labeled data may make rare statuses, flags, parts, and severities hard to characterize from examples alone.
- Multiple labels are serialized as semicolon-delimited text rather than normalized arrays.
- Multi-part claims can require evaluating more than one object part in a single conversation and image set.
- Some conversations contain irrelevant detail, multilingual text, or instructions embedded in user-provided content.
- Evidence requirements are natural-language categories; there is no explicit requirement ID attached to each claim.
- `claim_object` is present in test inputs, but issue type and object part must be inferred from conversation/image evidence in any later phase.
- Native image formats conflict with `.jpg` extensions; AVIF and WebP support may vary by tooling.
- Image sizes and aspect ratios vary substantially, and some labeled examples explicitly include blur, obstruction, wrong angle, wrong object, or non-original imagery.
- Case folder numbers cannot be treated as continuous row indices, especially in the test split.
- Free-text rationales are unique or nearly unique, so exact string distributions provide limited reusable structure.

## Questions and Ambiguities

1. Is the mixed native image encoding under `.jpg` filenames intentional and guaranteed in evaluation data?
2. Should `image_paths` always be resolved relative to `dataset/`, as the supplied files imply?
3. Is input row order the only required output-row alignment, given there is no claim/case ID column?
4. Can future claims contain more than three images or object classes outside car, laptop, and package?
5. For multi-part claims, should singular output fields represent the dominant issue, combine values, or follow another convention?
6. Are semicolon-delimited flags and supporting image IDs required in a canonical order?
7. How should generic `all` evidence requirements be combined with object-specific requirements when several apply?
8. Are `none` and `unknown` intentionally distinct for issue type, object part, and severity?
9. Are hidden `.DS_Store` files expected to remain in packaged evaluation datasets?
10. Does `valid_image=false` mean unreadable/irrelevant evidence, or can a technically readable image be invalid because it depicts the wrong object or non-original content?
