# Multi-Modal Damage Claim Verification System

This directory contains the Python implementation for the HackerRank Orchestrate Multi-Modal Evidence Review system. The system automatically verifies insurance damage claims using images, user claim transcripts, user history context, and minimum evidence requirements.

## System Architecture

The pipeline is designed as a modular, rate-limit resilient, and schema-compliant engine:

```
                  ┌──────────────────────┐
                  │ dataset/claims.csv   │
                  └──────────┬───────────┘
                             ▼
               ┌───────────────────────────┐
               │   Context Resolver        │
               │   - User history risk     │
               │   - Min evidence standard │
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │    Image Preprocessor     │
               │   - Always scale to 768   │
               │   - Compress to JPG (80%) │
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │  Rate-Resilient API VLM   │
               │   - Fallback model list   │
               │   - Structured JSON schema│
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │   Post-Process & Cleanup  │
               │   - Strict categorical map│
               │   - Safe fallback resolve │
               └─────────────┬─────────────┘
                             ▼
                  ┌──────────────────────┐
                  │      output.csv      │
                  └──────────────────────┘
```

### Key Components

1. **Context Lookup**:
   * Reads [`user_history.csv`](../dataset/user_history.csv) to inject historical risk flags (e.g., prior exaggerations or warnings).
   * Reads [`evidence_requirements.csv`](../dataset/evidence_requirements.csv) to supply the model with the minimum evidence standards for the target object type.

2. **Image Preprocessing & Downscaling**:
   * To prevent API rate limits (**Tokens Per Minute - TPM**), all images are automatically resized to a maximum dimension of `768x768` and compressed to 80% quality JPEG format. This reduces token consumption by over 90% while maintaining crisp visibility.

3. **Multi-Model Fallback Engine**:
   * The API router is configured with a fallback list of models (`gemini-2.0-flash`, `gemini-flash-latest` (1.5 Flash), and `gemini-2.5-flash`). If a model is rate-limited (429) or overloaded (503), it automatically attempts the next model, preventing pipeline failure.

4. **Schema Enforcement & Post-Processing**:
   * Enforces a structured JSON schema at the API generation layer.
   * Runs local regex and lookup validators to clean, map, and strictly align fields to permitted values in `problem_statement.md`.

---

## Getting Started

### 1. Installation
Initialize your virtual environment and install dependencies:
```bash
# Set up venv
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the root directory and add your Google AI Studio API key:
```env
GEMINI_API_KEY="your_api_key_here"
```

### 3. Running Predictions
To process `dataset/claims.csv` and generate predictions in `output.csv`:
```bash
python code/main.py
```

### 4. Running Evaluation
To evaluate accuracy on the sample labeled dataset:
```bash
python code/evaluation/main.py
```
This runs the pipeline on `sample_claims.csv` and generates the Markdown report at `code/evaluation/evaluation_report.md`.

---

## Output Fields Meaning

* `evidence_standard_met`: `true` if the visual evidence is sufficient to evaluate the claim; otherwise `false`.
* `risk_flags`: Semicolon-separated risk classifications (e.g. `blurry_image`, `user_history_risk`, `wrong_object`).
* `issue_type`: Category of damage visible (e.g. `dent`, `scratch`, `crack`, `glass_shatter`).
* `object_part`: Specific component affected (e.g. `rear_bumper`, `windshield`, `keyboard`, `seal`).
* `claim_status`: Final decision (`supported`, `contradicted`, or `not_enough_information`).
* `severity`: Estimated damage scale (`none`, `low`, `medium`, `high`, `unknown`).
