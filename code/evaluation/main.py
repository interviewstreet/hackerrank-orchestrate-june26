import sys
from pathlib import Path

# Append root folder paths to find core module engine imports effortlessly
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "code"))

from main import MultiModalReviewEngine

if __name__ == "__main__":
    print("Initiating verification system baseline evaluation run...")
    
    # Run pipeline using sample benchmark rows
    eval_engine = MultiModalReviewEngine(dataset_dir=str(root_dir / "dataset"))
    eval_engine.run_pipeline("sample_claims.csv", str(root_dir / "code" / "evaluation" / "sample_output.csv"))
    
    # Save the evaluation_report.md file requested by operational specs
    eval_engine.write_evaluation_markdown(str(root_dir / "code" / "evaluation" / "evaluation_report.md"))
    
    print("Evaluation steps complete! Verified benchmarks saved to code/evaluation/evaluation_report.md.")
