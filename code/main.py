import os
import csv
import time
import json
import hashlib
from core.processor import ClaimProcessor
from core.prompt_factory import build_vlm_system_prompt, build_user_analysis_prompt
from core.vlm_client import MultiModalVLMClient

# --- FEATURE A: LOCAL JSON CACHE ENGINE ---
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.cache"))
CACHE_FILE = os.path.join(CACHE_DIR, "vlm_cache.json")

def generate_cache_key(user_claim: str, image_paths: str) -> str:
    """Creates a unique deterministic MD5 hash string based on input evidence text and paths."""
    combined = f"{user_claim.strip()}|||{image_paths.strip()}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_to_cache(key: str, data_dict: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = load_cache()
    cache[key] = data_dict
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        print(f"  Cache write failed: {e}")

# INTERACTIVE TERMINAL DASHBOARD HELPERS 
def print_agent_header(row_num: int, user_id: str):
    print(f"\n[\033[1;36mAGENT ACTIVE\033[0m] ────► \033[1mProcessing Claim #{row_num} for User: {user_id}\033[0m")

def print_dashboard_step(step_type: str, message: str, symbol: str = "[+]", color_code: str = "32"):
    # color codes: 32 = green, 33 = yellow, 34 = blue, 35 = magenta
    print(f" ├── [\033[1;{color_code}m{symbol}\033[0m] {message}")

def print_agent_footer(status: str, source: str):
    color = "32" if "SUCCESS" in status or "HIT" in source else "33"
    print(f" └── [\033[1;{color}m{status}\033[0m] Handled via \033[1m{source}\033[0m")


def main():
    print("==================================================================")
    print("        LAUNCHING PRODUCTION AGENT RUN WITH CACHED DASHBOARD       ")
    print("==================================================================")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    dataset_dir = os.path.join(base_dir, "dataset")
    claims_csv = os.path.join(dataset_dir, "claims.csv")
    output_csv = os.path.join(base_dir, "output.csv")
    
    processor = ClaimProcessor(dataset_dir=dataset_dir)
    vlm_client = MultiModalVLMClient(provider="gemini")
    
    if not os.path.exists(claims_csv):
        print(f" ERROR: Input file claims.csv not found at: {claims_csv}")
        return
        
    vlm_cache = load_cache()
    
    headers = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    
    start_time = time.time()
    processed_count = 0
    cache_hits = 0
    api_calls = 0
    
    with open(claims_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
         
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        
        for row in reader:
            processed_count += 1
            user_id = row['user_id']
            
            # Draw header dashboard
            print_agent_header(processed_count, user_id)
            
            # Context Aggregation
            payload = processor.aggregate_claim_context(row)
            risk_summary = payload['user_profile'].get('history_summary', 'Clean History')
            print_dashboard_step("data", f"Database Context Enriched (Profile Summary: {risk_summary})", "[+]", "32")
            
            # Policy Compliance Check
            rules_count = len(payload.get('rules', []))
            print_dashboard_step("compliance", f"Compliance Schema Bound ({rules_count} business criteria rules mapped)", "[+]", "32")
            
            # Cache Verification Lookups
            cache_key = generate_cache_key(row['user_claim'], row['image_paths'])
            
            if cache_key in vlm_cache:
                cache_hits += 1
                print_dashboard_step("cache", f"Local Cache Hit! Bypassing outbound VLM network trip.", "[HIT]", "35")
                cached_data = vlm_cache[cache_key]
                
                writer.writerow({
                    "user_id": user_id,
                    "image_paths": row['image_paths'],
                    "user_claim": row['user_claim'],
                    "claim_object": row['claim_object'],
                    **cached_data
                })
                print_agent_footer("COMPLETED", "LOCAL JSON CACHE ENGINE")
                continue
                
            # Outbound VLM Run (Cache Miss)
            api_calls += 1
            print_dashboard_step("api", "Cache Miss. Dispatching Multi-Modal payload to Gemini 2.5 Flash API...", "[RUN]", "34")
            
            for img in payload['images']:
                img['path'] = os.path.join(dataset_dir, img['path'])
                
            system_prompt = build_vlm_system_prompt()
            user_prompt = build_user_analysis_prompt(payload)
            
            max_retries = 5
            backoff_delay = 4
            res = None
            
            for attempt in range(max_retries):
                try:
                    res = vlm_client.evaluate_claim(system_prompt, user_prompt, payload['images'])
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "503" in err_msg or "UNAVAILABLE" in err_msg:
                        print_dashboard_step("retry", f"Server busy. Retrying in {backoff_delay}s (Attempt {attempt+1}/{max_retries})...", "[WARN]", "33")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                    else:
                        break
            
            if res:
                out_fields = {
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
                }
                
                # Write to local file cache map
                save_to_cache(cache_key, out_fields)
                
                writer.writerow({
                    "user_id": user_id,
                    "image_paths": row['image_paths'],
                    "user_claim": row['user_claim'],
                    "claim_object": row['claim_object'],
                    **out_fields
                })
                print_agent_footer("CLAIM VERIFIED SUCCESSFULLY", "LIVE GEMINI 2.5 FLASH INFERENCE")
            else:
                print_dashboard_step("fail", "Failed all retry paths. Dropping fallback structural properties.", "[FAIL]", "31")
                writer.writerow({
                    "user_id": user_id,
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
                print_agent_footer("FALLBACK GENERATED", "RECOVERY MATRIX")
                
            # Co-operative pacing delay between API calls
            time.sleep(1)
                
    end_time = time.time()
    print("\n==================================================================")
    print(f" PRODUCTION RUN COMPLETE: Generated {processed_count} rows.")
    print(f"  ├── Cache Hits: {cache_hits} rows (Bypassed network)")
    print(f"  ├── API Invocations: {api_calls} calls (Cost: ${api_calls * 0.0013:.4f})")
    print(f"  └── Total runtime duration: {end_time - start_time:.2f} seconds.")
    print("==================================================================")

if __name__ == "__main__":
    main()