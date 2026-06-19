import os
import sys
import time
import json
import pandas as pd
from pathlib import Path

# Add project code directory to the front of sys.path to import from code/main.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import verify_claim, API_KEY


def run_evaluation():
    print("Starting system evaluation on sample dataset...")
    
    # Directory setup
    eval_dir = Path(__file__).parent
    base_dir = eval_dir.parent.parent
    dataset_dir = base_dir / "dataset"
    
    sample_csv = dataset_dir / "sample_claims.csv"
    user_history_csv = dataset_dir / "user_history.csv"
    evidence_req_csv = dataset_dir / "evidence_requirements.csv"
    report_file = eval_dir / "evaluation_report.md"
    
    if not sample_csv.exists():
        print(f"Error: sample_claims.csv not found at {sample_csv.resolve()}")
        return
        
    sample_df = pd.read_csv(sample_csv)
    user_history_df = pd.read_csv(user_history_csv) if user_history_csv.exists() else pd.DataFrame()
    evidence_req_df = pd.read_csv(evidence_req_csv) if evidence_req_csv.exists() else pd.DataFrame()
    
    if not API_KEY:
        print("Error: GEMINI_API_KEY is not configured in your environment. Cannot run evaluation.")
        return

    results = []
    total_rows = len(sample_df)
    print(f"Processing {total_rows} sample claims for evaluation...")
    
    start_time = time.time()
    consecutive_failures = 0
    quota_exhausted = False
    
    for idx, row in sample_df.iterrows():
        print(f"[{idx + 1}/{total_rows}] Evaluating sample claim for user_id: {row['user_id']}...")
        res = verify_claim(row, user_history_df, evidence_req_df, str(dataset_dir), quota_exhausted=quota_exhausted)
        results.append(res)
        
        # Check if this request was a fallback failure
        if "API quota exhausted" in res.get("evidence_standard_met_reason", "") or "Failed to contact visual review service" in res.get("evidence_standard_met_reason", ""):
            consecutive_failures += 1
            if consecutive_failures >= 3:
                if not quota_exhausted:
                    print("Detected 3 consecutive API failures. Setting quota_exhausted = True for remaining evaluation claims.")
                quota_exhausted = True
        else:
            consecutive_failures = 0
            
        # Free Tier Rate Limit: 15 Requests Per Minute (RPM) -> 1 request every 4.1 seconds
        if idx < total_rows - 1 and not quota_exhausted:
            time.sleep(4.1)
            
    total_latency = time.time() - start_time
    avg_latency = total_latency / total_rows if total_rows > 0 else 0
    
    # Analyze metrics
    correct_status = 0
    correct_part = 0
    correct_issue = 0
    correct_evidence = 0
    correct_severity = 0
    
    evaluation_rows = []
    
    for idx, pred in enumerate(results):
        actual = sample_df.iloc[idx]
        
        pred_status = pred["claim_status"]
        actual_status = str(actual["claim_status"]).strip().lower()
        status_match = pred_status == actual_status
        if status_match:
            correct_status += 1
            
        pred_part = pred["object_part"]
        actual_part = str(actual["object_part"]).strip().lower()
        part_match = pred_part == actual_part
        if part_match:
            correct_part += 1
            
        pred_issue = pred["issue_type"]
        actual_issue = str(actual["issue_type"]).strip().lower()
        issue_match = pred_issue == actual_issue
        if issue_match:
            correct_issue += 1
            
        pred_evidence = pred["evidence_standard_met"]
        actual_evidence = str(actual["evidence_standard_met"]).strip().lower()
        evidence_match = pred_evidence == actual_evidence
        if evidence_match:
            correct_evidence += 1
            
        pred_severity = pred["severity"]
        actual_severity = str(actual["severity"]).strip().lower()
        severity_match = pred_severity == actual_severity
        if severity_match:
            correct_severity += 1
            
        evaluation_rows.append({
            "user_id": pred["user_id"],
            "object": pred["claim_object"],
            "pred_status": pred_status,
            "actual_status": actual_status,
            "status_match": status_match,
            "pred_part": pred_part,
            "actual_part": actual_part,
            "part_match": part_match,
            "pred_issue": pred_issue,
            "actual_issue": actual_issue,
            "issue_match": issue_match
        })
        
    accuracy_status = correct_status / total_rows if total_rows > 0 else 0
    accuracy_part = correct_part / total_rows if total_rows > 0 else 0
    accuracy_issue = correct_issue / total_rows if total_rows > 0 else 0
    accuracy_evidence = correct_evidence / total_rows if total_rows > 0 else 0
    accuracy_severity = correct_severity / total_rows if total_rows > 0 else 0
    
    print("\n--- Evaluation Results ---")
    print(f"Total evaluated claims: {total_rows}")
    print(f"Claim Status Accuracy: {accuracy_status * 100:.2f}% ({correct_status}/{total_rows})")
    print(f"Object Part Accuracy: {accuracy_part * 100:.2f}% ({correct_part}/{total_rows})")
    print(f"Issue Type Accuracy: {accuracy_issue * 100:.2f}% ({correct_issue}/{total_rows})")
    print(f"Evidence Standard Met Accuracy: {accuracy_evidence * 100:.2f}% ({correct_evidence}/{total_rows})")
    print(f"Severity Accuracy: {accuracy_severity * 100:.2f}% ({correct_severity}/{total_rows})")
    print(f"Total Execution Time: {total_latency:.2f} seconds")
    print(f"Average Latency per Claim: {avg_latency:.2f} seconds")
    
    # Calculate approximate cost based on token assumptions
    # Gemini 2.5 Flash input: $0.075 / million tokens, output: $0.3 / million tokens
    # Approximate tokens per query (with image + transcript + requirements):
    # Image is ~258 tokens, prompt + user history + guidelines is ~1,000 tokens -> ~1,300 input tokens.
    # Output JSON is ~200 tokens.
    input_token_cost = (1300 / 1_000_000) * 0.075
    output_token_cost = (200 / 1_000_000) * 0.3
    cost_per_call = input_token_cost + output_token_cost
    est_sample_cost = cost_per_call * total_rows
    est_test_cost = cost_per_call * 45  # test file claims count
    
    # Generate the Markdown report
    report_content = f"""# System Evaluation Report

This report summarizes the performance evaluation of the multi-modal claims verification system against the sample dataset `sample_claims.csv`.

## Performance Metrics

| Metric | Accuracy | Correct Count | Total Count |
| :--- | :--- | :--- | :--- |
| **Claim Status (`claim_status`)** | {accuracy_status * 100:.1f}% | {correct_status} | {total_rows} |
| **Object Part (`object_part`)** | {accuracy_part * 100:.1f}% | {correct_part} | {total_rows} |
| **Issue Type (`issue_type`)** | {accuracy_issue * 100:.1f}% | {correct_issue} | {total_rows} |
| **Evidence Standard Met (`evidence_standard_met`)** | {accuracy_evidence * 100:.1f}% | {correct_evidence} | {total_rows} |
| **Severity (`severity`)** | {accuracy_severity * 100:.1f}% | {correct_severity} | {total_rows} |

## Operational Analysis

* **Model Used**: Gemini 2.5 Flash (`gemini-2.5-flash` endpoint)
* **Processing Speed**:
  * **Total latency (sample dataset)**: {total_latency:.2f} seconds
  * **Average latency per claim**: {avg_latency:.2f} seconds
  * **Free Tier rate limits**: Sleep delay of 4.1s added between calls to stay within the 15 RPM limit.
* **Token Usage Analysis**:
  * **Average input tokens per claim**: ~1,300 tokens (including base64 visual encoding and prompt context)
  * **Average output tokens per claim**: ~200 tokens (structured JSON response)
  * **Total input tokens (sample)**: ~{1300 * total_rows}
  * **Total output tokens (sample)**: ~{200 * total_rows}
* **Cost Analysis**:
  * **Pricing assumptions**: Gemini 2.5 Flash API costs $0.075 / 1M input tokens and $0.30 / 1M output tokens (standard pricing, though fully covered under Google AI Studio free tier).
  * **Approximate cost per call**: ${cost_per_call:.6f}
  * **Estimated cost for sample set**: ${est_sample_cost:.4f}
  * **Estimated cost for test set (45 rows)**: ${est_test_cost:.4f}

## Detailed Evaluation Table

| User ID | Object | Pred Status | Actual Status | Match? | Pred Part | Actual Part | Pred Issue | Actual Issue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    
    for row in evaluation_rows:
        match_str = "✅ Yes" if row["status_match"] else "❌ No"
        report_content += f"| {row['user_id']} | {row['object']} | {row['pred_status']} | {row['actual_status']} | {match_str} | {row['pred_part']} | {row['actual_part']} | {row['pred_issue']} | {row['actual_issue']} |\n"
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Evaluation report written to: {report_file.resolve()}")


if __name__ == "__main__":
    run_evaluation()
