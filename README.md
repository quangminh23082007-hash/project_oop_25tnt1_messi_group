# game-recsys-oop — models branch

Current delivery contract:
[Contracts v1.1 — reuse the trained models](docs/Contracts_v1_1_Pretrained_2026-11-06.html).
Run the existing BPR/TextBPR CPU bundles without retraining, rebuilding mappings
or downloading the text encoder. Python 3.10+ and NumPy are sufficient for serving.

Data integration follows the GitHub `data` branch at `c70e4f4f`, not the local
untracked Data prototypes. Its existing classes and file formats may remain;
catalog adapters join by raw game ID while the bundle owns its trained mapping.
The earlier [v1 training design](docs/Contracts_v1_Game_Recommender_2026-11-05.html)
and working training adapters are retained for optional future work.

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
model asset publication and the backend/UI connection. Rewriting every layer to
the earlier v1 training API is no longer a prerequisite for this serving delivery.

## Contents

- `src/contracts/`: shared v1 records, exceptions, interfaces and ranking helper.
- `src/models/`: public adapters, train-only review feature builder and provenance checks.
- `src/models/_array/`: preserved private numerical/legacy training engine.
- `configs/model_config.json`: example model defaults, not tuned production settings.
- `tests/test_team_models.py`: public v1 behavior with small independent fixtures.
- `tests/test_numerical_core.py`: gradient and numerical regression tests.

See [PRETRAINED_MODELS.md](docs/PRETRAINED_MODELS.md) for current delivery steps,
and [MODEL_HANDOFF.md](docs/MODEL_HANDOFF.md) for the optional training API.
The public training
adapters accept mapped datasets, use only train interactions, restore the best
validation checkpoint and invalidate failed fits. TextBPR requires aligned
`ItemFeatures` with matching mapping/train/review fingerprints.

The packaging configuration currently includes only shared contracts and models.
Expand it to include the other public layers when the team integrates them.

## Run the pretrained models without retraining

The original Kaggle CPU bundles serve recommendations with NumPy only. Contracts
v1.1 explicitly accepts their original 70/15/15 pair-split experiment and mappings.
This does not relabel them as models trained under the older v1 80/10/10 design.
Their recorded provenance and validation reports are in
[TRAINED_RUN_REFERENCE.json](docs/TRAINED_RUN_REFERENCE.json).

**Release status: assets prepared locally; not published yet.** The planned
[private release page](https://github.com/quangminh23082007-hash/project_oop_25tnt1_messi_group/releases/tag/pretrained-legacy-v1)
will provide `bpr-cpu-legacy-v1.zip` and `text-bpr-cpu-legacy-v1.zip` after upload.
Until then the link is not a working download. The repository owner should follow
[the publishing instructions](docs/PRETRAINED_MODELS.md). Repository read access
and GitHub sign-in are required; the weights do not need to be public.

After downloading the BPR ZIP from that release:

```powershell
python -m pip install -e .
python scripts/install_model.py --archive "C:\Users\HP\Downloads\bpr-cpu-legacy-v1.zip" --model bpr
python scripts/recommend.py --model bpr --user "!" --k 10
```

`!` is a source dataset user for a smoke check, not an application account. For
new users explicitly enable popularity fallback with `--allow-fallback`. Never
associate application accounts with dataset users by matching display names.

The installer verifies the complete archive against `configs/pretrained_models.json`
and then verifies the embedded weight checksum; it refuses to overwrite existing
models. If a verified bundle is already installed, proceed to the recommendation
command. TextBPR uses the same commands with `text-bpr` and its corresponding ZIP.

After publication, an alternative for teammates with authenticated GitHub CLI is:

```powershell
python scripts/install_model.py --download --model bpr
```

Load once in the backend, then reuse for requests:

```python
from dataclasses import asdict
from src.models.pretrained import PretrainedRecommender

recommender = PretrainedRecommender.load(
    "artifacts/models/bpr/cpu", allow_cold_start_fallback=True,
)
# Inside the backend's recommendation handler, supply real application history:
rows = recommender.recommend_games(
    raw_user_id="app:new-user", top_k=10, extra_seen=["570"],
)
response_json = [asdict(row) for row in rows]
```

Keep application IDs in a separate namespace from source dataset IDs. Pass
`catalog_lookup` to load current GameItem metadata for genre/price filters; old
bundles contain titles/images/categories but no reliable genres/prices. The UI
can retrieve bundled images via `get_catalog_metadata(game_id)` or its own catalog.
Do not call fit/train when serving these exports.

[PRETRAINED_MODELS.md](docs/PRETRAINED_MODELS.md) explains hosting, startup, CPU
requirements, filters and installation. [TEAM_COMPATIBILITY.md](docs/TEAM_COMPATIBILITY.md)
records the earlier v1 audit and identifies which work remains required under
the revised delivery scope. There is no
committed UI/HTTP endpoint in the branch trees inspected in that report; the
backend/UI owner must connect their actual handler to the loader above.

The preserved `python -m src --help` CLI and optional Kaggle helpers still implement
the earlier experiment protocol. They must not be used to claim new v1 results
without adapting their split, mapping, review and evaluation pipeline.
