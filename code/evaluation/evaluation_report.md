# System Evaluation Report

This report summarizes the performance evaluation of the multi-modal claims verification system against the sample dataset `sample_claims.csv`.

## Performance Metrics

| Metric | Accuracy | Correct Count | Total Count |
| :--- | :--- | :--- | :--- |
| **Claim Status (`claim_status`)** | 95.0% | 19 | 20 |
| **Object Part (`object_part`)** | 80.0% | 16 | 20 |
| **Issue Type (`issue_type`)** | 70.0% | 14 | 20 |
| **Evidence Standard Met (`evidence_standard_met`)** | 95.0% | 19 | 20 |
| **Severity (`severity`)** | 70.0% | 14 | 20 |

## Operational Analysis

* **Model Used**: Gemini 2.5 Flash (`gemini-2.5-flash` endpoint)
* **Processing Speed**:
  * **Total latency (sample dataset)**: 124.97 seconds
  * **Average latency per claim**: 6.25 seconds
  * **Free Tier rate limits**: Sleep delay of 4.1s added between calls to stay within the 15 RPM limit.
* **Token Usage Analysis**:
  * **Average input tokens per claim**: ~1,300 tokens (including base64 visual encoding and prompt context)
  * **Average output tokens per claim**: ~200 tokens (structured JSON response)
  * **Total input tokens (sample)**: ~26000
  * **Total output tokens (sample)**: ~4000
* **Cost Analysis**:
  * **Pricing assumptions**: Gemini 2.5 Flash API costs $0.075 / 1M input tokens and $0.30 / 1M output tokens (standard pricing, though fully covered under Google AI Studio free tier).
  * **Approximate cost per call**: $0.000158
  * **Estimated cost for sample set**: $0.0032
  * **Estimated cost for test set (45 rows)**: $0.0071

## Detailed Evaluation Table

| User ID | Object | Pred Status | Actual Status | Match? | Pred Part | Actual Part | Pred Issue | Actual Issue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| user_001 | car | supported | supported | ✅ Yes | rear_bumper | rear_bumper | dent | dent |
| user_002 | car | supported | supported | ✅ Yes | front_bumper | front_bumper | scratch | scratch |
| user_004 | car | supported | supported | ✅ Yes | windshield | windshield | crack | crack |
| user_007 | car | supported | supported | ✅ Yes | side_mirror | side_mirror | broken_part | broken_part |
| user_005 | car | contradicted | contradicted | ✅ Yes | quarter_panel | rear_bumper | dent | scratch |
| user_006 | car | contradicted | not_enough_information | ❌ No | headlight | headlight | none | unknown |
| user_003 | car | supported | supported | ✅ Yes | door | door | dent | dent |
| user_008 | car | contradicted | contradicted | ✅ Yes | hood | front_bumper | none | broken_part |
| user_009 | laptop | supported | supported | ✅ Yes | screen | screen | crack | crack |
| user_010 | laptop | supported | supported | ✅ Yes | hinge | hinge | broken_part | broken_part |
| user_011 | laptop | supported | supported | ✅ Yes | keyboard | keyboard | stain | stain |
| user_012 | laptop | supported | supported | ✅ Yes | lid | corner | dent | dent |
| user_018 | laptop | supported | supported | ✅ Yes | screen | screen | crack | crack |
| user_020 | laptop | contradicted | contradicted | ✅ Yes | trackpad | trackpad | scratch | none |
| user_015 | package | supported | supported | ✅ Yes | package_corner | package_corner | crushed_packaging | crushed_packaging |
| user_030 | package | supported | supported | ✅ Yes | seal | seal | torn_packaging | torn_packaging |
| user_031 | package | supported | supported | ✅ Yes | package_side | package_side | water_damage | water_damage |
| user_032 | package | not_enough_information | not_enough_information | ✅ Yes | contents | contents | unknown | unknown |
| user_033 | package | contradicted | contradicted | ✅ Yes | item | unknown | dent | unknown |
| user_034 | package | contradicted | contradicted | ✅ Yes | seal | seal | torn_packaging | none |
