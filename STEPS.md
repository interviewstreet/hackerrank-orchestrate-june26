# Verification Pipeline - Steps Guide

This document lists every step required to set up, optimize, evaluate, and execute the multi-modal claims verification system.

---

## Step 1: Initialize Virtual Environment
Initialize a local Python virtual environment `.venv` inside the `hackerrank-orchestrate-june26` repository directory:
```bash
# Navigate to the repo folder
cd hackerrank-orchestrate-june26

# Create the virtual environment
python -m venv .venv
```

---

## Step 2: Install Project Dependencies
Install the required data science, image processing, and HTTP request libraries:
```bash
# Activate the environment (Windows PowerShell)
.venv\Scripts\activate

# Install requirements
.venv\Scripts\pip install -r requirements.txt
```
*Dependencies installed:* `pandas`, `pillow`, `python-dotenv`, and `requests`.

---

## Step 3: Configure environment variables (.env)
Create a `.env` file in the root repository directory and populate it with your Google AI Studio API key:
```env
GEMINI_API_KEY="your_api_key_here"
```

---

## Step 4: Run System Evaluation
Validate the accuracy of the multi-modal prompt instructions against the 20 ground-truth labeled examples in the sample dataset:
```bash
.venv\Scripts\python.exe code/evaluation/main.py
```
*Outputs generated:* 
* Prints evaluation scores (Accuracy of Status, Part, Issue, Evidence, and Severity).
* Writes a detailed markdown report to [`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md).

---

## Step 5: Run Claims Prediction
Process the final test dataset (`dataset/claims.csv`) containing 44 claims and write the outputs:
```bash
.venv\Scripts\python.exe code/main.py
```
*Outputs generated:*
* Writes predictions to the root [`output.csv`](output.csv).

---

## Step 6: Sync Output File
Duplicate the root output file to the dataset folder to ensure all submissions match the expected evaluator location:
```bash
python -c "import shutil; shutil.copy2('output.csv', 'dataset/output.csv')"
```

---

## Step 7: Bundle Submission Package
Package the runnable code, documentation, and evaluation scripts into `code.zip` while ignoring the `.venv` and Python cache files:
```bash
.venv\Scripts\python.exe C:\Users\sathv\.gemini\antigravity-ide\brain\3f10c411-3f3b-4f1d-bce2-bcd51834efcc\scratch\zip_solution.py
```
*Outputs generated:*
* Generates a clean [`code.zip`](code.zip) file at the repository root containing `code/` and documentation.

---

## Calculated System Accuracies
The latest evaluation run on `dataset/sample_claims.csv` yielded the following accuracy metrics:

| Metric | Calculated Accuracy | Correct Count | Total Count |
| :--- | :--- | :--- | :--- |
| **Claim Status (`claim_status`)** | **95.0%** | 19 | 20 |
| **Evidence Standard Met (`evidence_standard_met`)** | **95.0%** | 19 | 20 |
| **Object Part (`object_part`)** | **80.0%** | 16 | 20 |
| **Issue Type (`issue_type`)** | **70.0%** | 14 | 20 |
| **Severity (`severity`)** | **70.0%** | 14 | 20 |

*Note: These metrics are evaluated against the 20 ground-truth samples in `sample_claims.csv`.*

---

## Technical Flow inside `main.py`
For each processed claim row, the script executes these steps automatically:
1. **Context Mapping**: Loads user history profile and matches objects to evidence rules.
2. **Visual Optimization**: Automatically resizes every loaded image to a maximum of `768x768` pixels to save tokens and prevent rate limit blocks.
3. **Structured VLM Prompting**: Packs the base64-encoded image and claim transcripts into a single request, enforcing a JSON schema returned from the Gemini API.
4. **Resilient Routing**: Attempts fallback models (`gemini-flash-lite-latest`, `gemini-2.0-flash`, `gemini-flash-latest`, `gemini-2.5-flash`) if rate limits occur.
5. **Fail-Safe Quota Handler**: If the API key triggers 3 consecutive errors (e.g. daily quota reached), it bypasses further calls and runs a fast-path fallback to generate valid structured outputs.

