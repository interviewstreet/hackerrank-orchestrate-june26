# Claim Verifier

Minimal Python solution for the HackerRank Orchestrate challenge.

## What it does

- Reads `dataset/claims.csv`, `dataset/user_history.csv`, and `dataset/evidence_requirements.csv`
- Resolves local image paths under `dataset/images/...`
- Calls Gemini with the claim chat, relevant evidence requirements, and submitted images
- Normalizes the model output into the exact `output.csv` schema
- Optionally runs a gated judge pass on harder rows
- Evaluates two strategies on `dataset/sample_claims.csv`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r code/requirements.txt
   ```

3. Copy `code/.env.example` to `.env` at the repo root or to `code/.env`, then fill in `GEMINI_API_KEY`.

## Prompt and model config

- Prompt files live under `code/prompts/` and must be plain `.txt` files with a versioned name such as `analysis_v1.txt` or `judge_v1.txt`.
- Final prediction config lives in `code/configs/prediction_config.yaml`.
- Evaluation config lives in `code/configs/evaluation_config.yaml`.
- `prediction_config.yaml` controls:
  - where final predictions are written
  - which single strategy is used for `output.csv`
- `evaluation_config.yaml` controls:
  - where evaluation outputs are written
  - how many sample rows to evaluate
  - the random seed used for sample-set selection
  - which strategies to compare

To compare multiple models or prompt versions, add more entries under `strategies` in `code/configs/evaluation_config.yaml`.

## Run predictions

```bash
python code/main.py
```

This writes `output.csv` inside a timestamped run folder under the configured prediction `output_dir`.

## Run evaluation

```bash
python code/evaluation/main.py
```

This writes, inside a timestamped evaluation run folder:

- `evaluation/evaluation_report.md`
- `evaluation/report.html`
- `runtime_traces.jsonl`
- `llm_calls.jsonl`
- `config/` with the exact config file used for that run
- `strategies/<strategy_name>/output.csv` for every evaluated strategy, using the final required output schema

Evaluation uses the seeded sample subset defined in `code/configs/evaluation_config.yaml`, and every listed strategy runs against the same sampled rows for a fair comparison.

Each run creates a separate timestamped folder inside the configured `output_dir`, for example `outputs/prediction_runs/prediction_20260619_145500/`. That folder stores the generated outputs plus a `config/` snapshot of the YAML file used for the run.

- `runtime_traces.jsonl` logs claim-level progress, including which sample or claim is being processed, which strategy is running, and whether each claim finished successfully or failed.
- `llm_calls.jsonl` stores one JSON record for each real Gemini API call, including the exact prompt text sent, the image references included, and the raw text returned by the model.

## Notes

- The code is intentionally small and readable rather than heavily abstracted.
- Missing image files are handled as `not_enough_information`.
- Unexpected API or parsing failures raise clear exceptions so they are visible during development.
