import os
import csv
import time
from core.processor import ClaimProcessor
from core.prompt_factory import build_vlm_system_prompt, build_user_analysis_prompt
from core.vlm_client import MultiModalVLMClient

def main():
    print("==================================================")
    print("        LAUNCHING PRODUCTION CLAIMS RUN           ")
    print("==================================================")
    
    # Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    dataset_dir = os.path.join(base_dir, "dataset")
    claims_csv = os.path.join(dataset_dir, "claims.csv")
    output_csv = os.path.join(base_dir, "output.csv") # Output goes in root or dataset as requested
    
    # Initialize engines
    processor = ClaimProcessor(dataset_dir=dataset_dir)
    vlm_client = MultiModalVLMClient(provider="gemini")
    
    if not os.path.exists(claims_csv):
        print(f"Input file claims.csv not found at: {claims_csv}")
        return
        
    # Open output file and write strict schema headers in exact specified order
    headers = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    
    start_time = time.time()
    processed_count = 0
    
    with open(claims_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
         
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        
        for row in reader:
            processed_count += 1
            print(f"Processing row {processed_count} for user: {row['user_id']}...")
            
            # Aggregate lookups
            payload = processor.aggregate_claim_context(row)
            
            # Reconstruct absolute paths for image files
            for img in payload['images']:
                img['path'] = os.path.join(dataset_dir, img['path'])
                
            # Compile Prompts
            system_prompt = build_vlm_system_prompt()
            user_prompt = build_user_analysis_prompt(payload)
            
            # Robust Retry Loop for Production Stability
            max_retries = 5
            backoff_delay = 4
            res = None
            
            for attempt in range(max_retries):
                try:
                    res = vlm_client.evaluate_claim(system_prompt, user_prompt, payload['images'])
                    break  # Success! Break out of retry loop
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "503" in err_msg or "UNAVAILABLE" in err_msg:
                        print(f"  ⚠️ Server busy or Rate limit hit. Backing off for {backoff_delay}s (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2  # Exponential backoff
                    else:
                        print(f"  Non-quota error encountered: {e}")
                        break
            
            # Write row data strictly aligned to requested output types
            if res:
                writer.writerow({
                    "user_id": row['user_id'],
                    "image_paths": row['image_paths'],
                    "user_claim": row['user_claim'],
                    "claim_object": row['claim_object'],
                    "evidence_standard_met": str(res.evidence_standard_met).lower(),
                    "evidence_standard_met_reason": res.evidence_standard_met_reason,
                    "risk_flags": res.risk_flags if res.risk_flags else "none",
                    "issue_type": res.issue_type,
                    "object_part": res.object_part,
                    "claim_status": res.claim_status,
                    "claim_status_justification": res.claim_status_justification,
                    "supporting_image_ids": res.supporting_image_ids if res.supporting_image_ids else "none",
                    "valid_image": str(res.valid_image).lower(),
                    "severity": res.severity
                })
            else:
                print(f"  Failed all retries for user {row['user_id']}. Recording fallback row.")
                writer.writerow({
                    "user_id": row['user_id'],
                    "image_paths": row['image_paths'],
                    "user_claim": row['user_claim'],
                    "claim_object": row['claim_object'],
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "API execution failure",
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "System API exception during processing.",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "unknown"
                })
                
            # Keep a small safety pacing delay between production items
            time.sleep(1)
                
    end_time = time.time()
    print("\n==================================================")
    print(f" PRODUCTION RUN COMPLETE: Generated {processed_count} rows.")
    print(f" Total runtime duration: {end_time - start_time:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    main()