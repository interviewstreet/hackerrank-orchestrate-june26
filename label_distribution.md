# Label Distribution and Decision Pattern Analysis

Analysis timestamp: 2026-06-19 (Asia/Calcutta)

## Scope and Method

This analysis uses only the 20 labeled rows in `dataset/sample_claims.csv` and the 11 rows in `dataset/evidence_requirements.csv`. It describes observed associations and dataset-creator hypotheses; it does not define implementation logic, prompts, or architecture.

Examples are identified as `case_NNN`, derived from each row's `image_paths`. Percentages for general labels use all 20 sample rows. Object-part percentages use the number of rows for that object (car 8, laptop 6, package 6). Atomic risk-flag percentages use 20 rows as the denominator; a multi-flag row contributes once to every listed flag.

## 1. Statistics

| Statistic | Value |
|---|---:|
| Labeled rows | 20 |
| Car rows | 8 (40.0%) |
| Laptop rows | 6 (30.0%) |
| Package rows | 6 (30.0%) |
| One-image rows | 11 (55.0%) |
| Two-image rows | 9 (45.0%) |
| Rows with `risk_flags=none` | 11 (55.0%) |
| Rows with one or more risk flags | 9 (45.0%) |
| Evidence standard met | 18 (90.0%) |
| Valid image | 18 (90.0%) |

## 2. Label Frequencies

### `claim_status`

| Value | Count | Percentage | Example rows |
|---|---:|---:|---|
| supported | 13 | 65.0% | case_001 (rear-bumper dent), case_009 (screen crack), case_015 (crushed package corner) |
| contradicted | 5 | 25.0% | case_005 (severity mismatch), case_014 (no trackpad damage), case_019 (wrong object) |
| not_enough_information | 2 | 10.0% | case_006 (headlight not shown), case_018 (contents unclear) |

By object, car has 5 supported, 2 contradicted, and 1 insufficient row; laptop has 5 supported and 1 contradicted row; package has 3 supported, 2 contradicted, and 1 insufficient row.

### `issue_type`

| Value | Count | Percentage | Example rows |
|---|---:|---:|---|
| dent | 3 | 15.0% | case_001, case_007, case_012 |
| crack | 3 | 15.0% | case_003, case_009, case_013 |
| broken_part | 3 | 15.0% | case_004, case_008, case_010 |
| unknown | 3 | 15.0% | case_006, case_018, case_019 |
| scratch | 2 | 10.0% | case_002, case_005 |
| none | 2 | 10.0% | case_014, case_020 |
| stain | 1 | 5.0% | case_011 |
| crushed_packaging | 1 | 5.0% | case_015 |
| torn_packaging | 1 | 5.0% | case_016 |
| water_damage | 1 | 5.0% | case_017 |

Status behavior by issue is informative but based on few rows: all dent and crack rows are supported; `none` appears only on contradicted rows; `unknown` appears on the two insufficient rows and one wrong-object contradiction. `broken_part` and `scratch` occur in both supported and contradicted outcomes, showing that issue type alone does not determine status.

### `severity`

| Value | Count | Percentage | Example rows |
|---|---:|---:|---|
| medium | 11 | 55.0% | case_001, case_009, case_017 |
| low | 4 | 20.0% | case_002, case_005, case_012, case_019 |
| unknown | 2 | 10.0% | case_006, case_018 |
| none | 2 | 10.0% | case_014, case_020 |
| high | 1 | 5.0% | case_008 |

All 11 medium rows are supported. Low severity is split between 2 supported and 2 contradicted rows. Both `none` rows are contradicted, both `unknown` rows are insufficient, and the single high row is contradicted because severe visible front-end damage differs from the claimed hood scratch. These are observed sample associations, not proof that severity determines status.

### `object_part` by object

Percentages below are within each object group.

#### Car (`n=8`)

| Requested part | Count | Percentage |
|---|---:|---:|
| front_bumper | 2 | 25.0% |
| rear_bumper | 2 | 25.0% |
| door | 1 | 12.5% |
| hood | 0 | 0.0% |
| windshield | 1 | 12.5% |
| side_mirror | 1 | 12.5% |
| headlight | 1 | 12.5% |
| taillight | 0 | 0.0% |
| fender | 0 | 0.0% |
| quarter_panel | 0 | 0.0% |
| body | 0 | 0.0% |
| unknown | 0 | 0.0% |

Notably, case_008 claims a hood scratch but is labeled `front_bumper` because that is the visible damaged part. This indicates the output can describe observed evidence rather than merely repeat the claim.

#### Laptop (`n=6`)

| Requested part | Count | Percentage |
|---|---:|---:|
| screen | 2 | 33.3% |
| keyboard | 1 | 16.7% |
| trackpad | 1 | 16.7% |
| hinge | 1 | 16.7% |
| lid | 0 | 0.0% |
| corner | 1 | 16.7% |
| port | 0 | 0.0% |
| base | 0 | 0.0% |
| body | 0 | 0.0% |
| unknown | 0 | 0.0% |

#### Package (`n=6`)

| Requested part | Count | Percentage |
|---|---:|---:|
| box | 0 | 0.0% |
| package_corner | 1 | 16.7% |
| package_side | 1 | 16.7% |
| seal | 2 | 33.3% |
| label | 0 | 0.0% |
| contents | 1 | 16.7% |
| item | 0 | 0.0% |
| unknown | 1 | 16.7% |

Case_019 claims a crushed shipping box but is labeled `unknown` because the visible object does not match the claimed package.

## 3. Risk Flag Frequencies

### Atomic flags

| Flag | Count | Row frequency | Example rows |
|---|---:|---:|---|
| none | 11 | 55.0% | case_001, case_009, case_015 |
| manual_review_required | 7 | 35.0% | case_005, case_017, case_018 |
| user_history_risk | 6 | 30.0% | case_005, case_014, case_019 |
| damage_not_visible | 4 | 20.0% | case_006, case_014, case_018, case_020 |
| claim_mismatch | 3 | 15.0% | case_005, case_008, case_019 |
| blurry_image | 1 | 5.0% | case_007 |
| cropped_or_obstructed | 1 | 5.0% | case_018 |
| non_original_image | 1 | 5.0% | case_008 |
| text_instruction_present | 1 | 5.0% | case_020 |
| wrong_angle | 1 | 5.0% | case_006 |
| wrong_object | 1 | 5.0% | case_019 |

The counts correctly split compound cells. For example, case_005 contributes one count each to `claim_mismatch`, `user_history_risk`, and `manual_review_required`.

### Flag behavior

- `none` occurs only on supported rows in this sample, but supported case_007 and case_017 show that support does not require `none`.
- `user_history_risk` appears in five contradicted rows and one supported row (case_017). History therefore adds review context but does not override clear visual support.
- `manual_review_required` appears in all five contradicted rows except case_006, plus supported case_017 and insufficient case_018. It is not synonymous with any one status.
- `damage_not_visible` spans contradiction and insufficiency. When the claimed area is visible and undamaged (case_014, case_020), the result is contradicted; when the relevant area cannot be inspected (case_006, case_018), the result is insufficient.
- `blurry_image` does not force insufficiency: case_007 is supported because the second image clearly shows the dent.
- `text_instruction_present` is treated as a risk to ignore, not as evidence (case_020).

## 4. Decision Patterns

### When `claim_status=supported`

All 13 supported rows have `evidence_standard_met=true` and `valid_image=true`. Their common pattern is affirmative visual alignment among the claimed object/part, the visible condition, and the described issue.

- Directly visible damage: case_001 shows the rear-bumper dent; case_009 shows a screen crack; case_015 shows a crushed package corner.
- Correct part visibility: reasons explicitly identify the relevant bumper, glass, mirror, door, screen, hinge, keyboard, corner, package surface, or seal.
- Multi-image complementarity: case_002 uses a full front view plus a close-up; case_010 uses full-laptop context plus hinge detail; case_016 uses torn-seal detail plus full-package context.
- One sufficient image can overcome another weak image: case_007 remains supported despite `blurry_image` because `img_2` clearly shows the door dent.
- History risk does not defeat clear evidence: case_017 is supported while carrying `user_history_risk;manual_review_required`.

### When `claim_status=contradicted`

All 5 contradicted rows have `evidence_standard_met=true`: the images are considered sufficient to decide, but the visible evidence conflicts with the claim.

- Severity/extent mismatch: case_005 shows only a small rear-bumper scratch rather than the claimed bad damage.
- Different visible issue or part: case_008 shows severe front-end/front-bumper damage rather than a hood scratch.
- Claimed damage absent on an inspectable part: case_014 clearly shows the trackpad area without physical damage; case_020 shows an intact seal rather than torn-open packaging.
- Wrong object: case_019 shows a visibly creased/dented object that is not the claimed shipping box.

Contradiction is therefore not the same as poor evidence. Four contradicted rows have `valid_image=true`; case_008 is the exception (`valid_image=false`) but is still visually decisive enough to establish mismatch.

### When `claim_status=not_enough_information`

Both insufficient rows have `evidence_standard_met=false`, `issue_type=unknown`, `severity=unknown`, `supporting_image_ids=none`, and `damage_not_visible`.

- Missing relevant part/wrong angle: case_006 does not show the claimed headlight. The image itself is valid, but it cannot verify the target.
- Cropped or obstructed evidence: case_018 does not clearly show the expected contents or enough of the opened package. The images are labeled invalid and cannot verify missing contents.

The key distinction from contradiction is inspectability. If the claimed area is visible enough and the claimed damage is absent, the sample uses `contradicted`; if the area/evidence cannot be inspected adequately, it uses `not_enough_information`.

## 5. Evidence Requirement Findings

### Requirements most visibly associated with decisions

| Requirement | Sample evidence and observed effect |
|---|---|
| `REQ_GENERAL_OBJECT_PART` | Fundamental across all rows. Clear claimed-part visibility supports case_001, case_009, and case_015; missing target visibility causes insufficiency in case_006 and case_018. |
| `REQ_REVIEW_TRUST` | Relevance and grounding distinguish supported evidence from non-original or wrong-object evidence in case_008 and case_019. |
| `REQ_GENERAL_MULTI_IMAGE` | Case_007 shows that one clear relevant image can satisfy a two-image set despite another blurry image. Case_018 shows that multiple images do not help when none adequately shows the contents. |
| `REQ_CAR_BODY_PANEL` | Clear panel/bumper angles support case_001, case_002, and case_007. The same evaluability permits contradiction for severity mismatch in case_005. |
| `REQ_CAR_GLASS_LIGHT_MIRROR` | Visible glass/mirror damage supports case_003 and case_004; failure to show the headlight drives insufficiency in case_006. |
| `REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD` | Visible cracks/staining support case_009, case_011, and case_013. A clear, undamaged trackpad contradicts case_014 rather than yielding insufficiency. |
| `REQ_LAPTOP_BODY_HINGE_PORT` | Context plus detail supports the hinge in case_010 and corner dent in case_012. |
| `REQ_PACKAGE_EXTERIOR` | Visible corner/seal damage supports case_015 and case_016; a clearly intact seal contradicts case_020. |
| `REQ_PACKAGE_LABEL_OR_STAIN` | A visible stained surface supports the water-damage claim in case_017. |
| `REQ_PACKAGE_CONTENTS` | Case_018 fails because the opened package and expected contents area are not clear enough to assess a missing item. |
| `REQ_CAR_IDENTITY_OR_SIDE` | Orientation/context matters in case_006 (wrong view) and grounding matters in mismatch cases, though the sample does not explicitly name requirement IDs per row. |

### Supported versus insufficient evidence

The requirements govern whether the evidence can answer the claim, not whether the answer is favorable. Satisfying a requirement can produce either support (damage visible) or contradiction (inspectable part but claim absent/mismatched). Failing a visibility requirement produces the two insufficient rows.

No explicit requirement ID is attached to sample rows, so mappings above are semantic comparisons between `claim_object`, issue/part, reasons, and requirement text. They are evidence-backed interpretations rather than recorded joins.

## 6. Output Field Behavior

| Field | Observed values | Observed pattern / inferred behavior |
|---|---|---|
| `evidence_standard_met` | true 18; false 2 | Indicates whether evidence is sufficient to evaluate. It is true for every supported and contradicted row, false for both insufficient rows. |
| `evidence_standard_met_reason` | 20 unique free-text values | Explains visibility, relevance, angle, context, and whether the claimed condition can be inspected. It can describe sufficient evidence that contradicts a claim. |
| `risk_flags` | 11 atomic values including `none` | Semicolon-delimited, additive concerns from image quality, mismatch, provenance, embedded text, and history. Flags provide context and do not independently determine status. |
| `issue_type` | dent, crack, broken_part, unknown, scratch, none, stain, crushed_packaging, torn_packaging, water_damage | Appears to describe the evidence-grounded issue. `none` marks visible claimed areas without damage; `unknown` marks unresolvable or wrong-object cases; mismatches can output the actually visible issue (case_008). |
| `object_part` | 16 observed values | Usually identifies the claimed/verified part, but can switch to the visibly damaged part during contradiction (case_008) or `unknown` for wrong object (case_019). |
| `claim_status` | supported, contradicted, not_enough_information | Separates aligned evidence, decisive conflicting evidence, and evidence that cannot answer the claim. |
| `claim_status_justification` | 20 unique free-text values | Grounds status in visible evidence and sometimes adds history/manual-review context. It distinguishes absence from inability to inspect. |
| `supporting_image_ids` | `img_1` 15; `img_2` 2; `img_1;img_2` 1; `none` 2 | Names images that substantiate the decision, including contradictory decisions. Both insufficient rows use `none`; not every submitted image is automatically listed. |
| `valid_image` | true 18; false 2 | Not equivalent to evidence sufficiency. Case_006 is valid but insufficient (wrong view); case_008 is invalid/non-original but still sufficient to establish contradiction. |
| `severity` | medium, low, unknown, none, high | Appears evidence-grounded. `none` accompanies visible absence and contradiction; `unknown` accompanies insufficient evidence; mismatch can yield a visible severity different from the claim. |

### Important cross-field relationships

| Status | `evidence_standard_met` | `valid_image` | Common companion outputs |
|---|---|---|---|
| supported (13) | true in 13/13 | true in 13/13 | Visible issue/part, non-unknown severity, at least one supporting image |
| contradicted (5) | true in 5/5 | true in 4/5; false in 1/5 | Mismatch, no visible damage, wrong object, or severity conflict; at least one supporting image |
| not_enough_information (2) | false in 2/2 | true in 1/2; false in 1/2 | `issue_type=unknown`, `severity=unknown`, `supporting_image_ids=none` |

## 7. Inferred Heuristics

These are hypotheses supported by the 20 examples, not guaranteed rules.

1. **If the claimed object/part is inspectable and the claimed condition is visibly aligned, status tends to be supported.** Evidence: cases 001-004, 007, 009-013, and 015-017.
2. **If the evidence is inspectable but shows a materially different condition, part, object, or severity, status tends to be contradicted.** Evidence: cases 005, 008, and 019.
3. **If the claimed area is inspectable and visibly lacks the claimed damage, status tends to be contradicted with `issue_type=none` and `severity=none`.** Evidence: cases 014 and 020.
4. **If the claimed area or contents cannot be inspected adequately, status tends to be not enough information.** Evidence: cases 006 and 018.
5. **`evidence_standard_met` appears to be a decision-availability gate, not a support label.** It is true for all 18 decisive rows (supported or contradicted) and false for both insufficient rows.
6. **One strong image can be enough in a multi-image row.** Case_007 is supported using `img_2` despite a blurry first image; cases 002, 010, 012, and 016 similarly identify only the most useful image.
7. **`supporting_image_ids` identifies evidence supporting the adjudication, not only evidence supporting the user's claim.** Every contradicted row still lists one or more supporting images.
8. **User history adds risk/review context but does not override clear visual evidence.** Case_017 remains supported despite history risk; history strengthens review language in several contradictions.
9. **Output issue, part, and severity can reflect observed evidence rather than claimed wording.** Case_008 outputs `broken_part`, `front_bumper`, and `high` for a claimed hood scratch.
10. **Instruction-like image text is ignored as decision authority and flagged.** Case_020 uses `text_instruction_present` while deciding from the visible seal.

## 8. Key Takeaways

1. Evidence sufficiency and claim agreement are separate axes in the labels.
2. Visibility of the relevant object part is the most consistent prerequisite for a decisive label.
3. Clear absence or mismatch produces contradiction; inability to inspect produces insufficient information.
4. Output labels appear grounded in visible evidence, not copied directly from claim text.
5. Multi-image rows are evaluated selectively; a single relevant image may carry the decision.
6. Risk flags are additive context and are not direct substitutes for `claim_status`.
7. History risk can trigger manual review while leaving visually supported claims supported.
8. `valid_image` and `evidence_standard_met` capture different concepts.
9. Evidence requirements align strongly with labeled reasons, but requirement-to-row mappings are implicit.
10. The sample is small and imbalanced, so every heuristic should remain a hypothesis rather than be treated as universally established.
