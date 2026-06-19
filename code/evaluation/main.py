import sys
import os
import csv
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.processor import ClaimProcessor
from core.prompt_factory import build_vlm_system_prompt, build_user_analysis_prompt
from core.vlm_client import MultiModalVLMClient

def run_evaluation():
    print("==================================================")
    print("          RUNNING SAMPLE CLAIMS EVALUATION        ")
    print("==================================================")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    dataset_dir = os.path.join(base_dir, "dataset")
    sample_claims_csv = os.path.join(dataset_dir, "sample_claims.csv")
    
    processor = ClaimProcessor(dataset_dir=dataset_dir)
    vlm_client = MultiModalVLMClient(provider="gemini")
    
    if not os.path.exists(sample_claims_csv):
        print(f"Could not find sample_claims.csv at: {sample_claims_csv}")
        return

    success_count = 0
    total_count = 0
    
    with open(sample_claims_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_count += 1
            user_id = row['user_id']
            expected_status = row['claim_status']
            print(f"\n[{total_count}] Evaluating claim for {user_id} ({row['claim_object']})...")
            
            payload = processor.aggregate_claim_context(row)
            
            for img in payload['images']:
                img['path'] = os.path.join(dataset_dir, img['path'])
            
            system_prompt = build_vlm_system_prompt()
            user_prompt = build_user_analysis_prompt(payload)
            
            # Retry loop with exponential backoff to handle rate limits
            max_retries = 5
            backoff_delay = 4  # seconds
            evaluation_result = None
            
            for attempt in range(max_retries):
                try:
                    evaluation_result = vlm_client.evaluate_claim(system_prompt, user_prompt, payload['images'])
                    break  # Success! Break out of retry loop
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f" Rate limit hit (429). Backing off for {backoff_delay}s (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2  # Double the wait time for the next attempt
                    else:
                        print(f" Non-quota error encountered: {e}")
                        break
            
            if evaluation_result:
                predicted_status = evaluation_result.claim_status
                is_match = (predicted_status == expected_status)
                
                if is_match:
                    success_count += 1
                    match_status = "MATCH"
                else:
                    match_status = "MISMATCH"
                    
                print(f"  -> Predicted Status: '{predicted_status}' | Expected: '{expected_status}' [{match_status}]")
                print(f"  -> Justification: {evaluation_result.claim_status_justification}")
                print(f"  -> Part Identified: {evaluation_result.object_part} | Severity: {evaluation_result.severity}")
            else:
                print(f"  Skipping row due to persistent API failures.")
            
            # Cooperative throttling: insert a minor safety delay between items
            time.sleep(5)

    accuracy = (success_count / total_count) * 100 if total_count > 0 else 0
    print("\n==================================================")
    print(f" EVALUATION FINISHED: {success_count}/{total_count} Passed Correctly")
    print(f" Baseline Dataset Accuracy: {accuracy:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()