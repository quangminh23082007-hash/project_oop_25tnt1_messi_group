# game-recsys-oop — models branch

Popularity, BPR and TextBPR implementing the team's
[Contracts v1 (2026-11-05)](docs/Contracts_v1_Game_Recommender_2026-11-05.html).
Python 3.10+ and NumPy are sufficient for the public CPU models and model tests.

## Install and verify this contribution

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_team_models.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_numerical_core.py -v
```

On Linux/macOS use `python3 -m venv .venv` and `.venv/bin/python`.
Tests run without the future Data layer, external datasets, GPU or model downloads.
This branch provides the model contribution; a complete application also needs
the team's v1 Data, Evaluation and Service layers.

## Contents

- `src/contracts/`: shared v1 records, exceptions, interfaces and ranking helper.
- `src/models/`: public adapters, train-only review feature builder and provenance checks.
- `src/models/_array/`: preserved private numerical/legacy training engine.
- `configs/model_config.json`: example model defaults, not tuned production settings.
- `tests/test_team_models.py`: public v1 behavior with small independent fixtures.
- `tests/test_numerical_core.py`: gradient and numerical regression tests.

See [MODEL_HANDOFF.md](docs/MODEL_HANDOFF.md) for the API, required Data handoff,
exact model-branch file scope and integration responsibilities. The public
adapters accept mapped datasets, use only train interactions, restore the best
validation checkpoint and invalidate failed fits. TextBPR requires aligned
`ItemFeatures` with matching mapping/train/review fingerprints.

The packaging configuration currently includes only shared contracts and models.
Expand it to include the other public layers when the team integrates them.

## Existing model artifacts and Kaggle tools

Trained weights and datasets are separate artifacts, not ordinary Git source.
If the optional importer is included, restore compact legacy CPU bundles with:

```powershell
python scripts/install_model.py --archive "C:\Users\HP\Downloads\both_models_and_reports.zip"
```

The importer checks weight checksums and refuses to overwrite an existing bundle.
No GPU or Qwen weights are needed to score those compact legacy CPU bundles.
They retain their original mapping and evaluation protocol; they are not v1-trained
models. The old application integration must be updated separately for v1.

`python -m src --help` exposes the preserved legacy offline CLI. Optional Kaggle
helper files can reproduce the earlier workflow, but they do not yet implement
the new v1 split/review/evaluation protocol. Keep new v1 runs separate from old
artifacts and never relabel their fingerprints to bypass compatibility checks.
