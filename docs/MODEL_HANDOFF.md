# Models branch handoff — Contracts v1 (2026-11-05)

Source of truth: [team contract](Contracts_v1_Game_Recommender_2026-11-05.html).
This contribution implements the Model boundary and proposes the shared v1
records/interfaces it imports. The team should use these same shared definitions;
Data, Evaluation and Service must import/re-export them, not define duplicates.
This is not a claim that the other branches already implement v1.

## Files for this contribution

Required: `src/models/` (including the whole private `_array` engine),
`src/contracts/`, `src/__init__.py`, `src/__main__.py`, `src/cli.py`,
`tests/test_team_models.py`, `tests/test_numerical_core.py`, `pyproject.toml`,
`.gitignore`, `README.md`, `docs/`, `configs/model_config.json`.

Optional legacy model importer: `scripts/install_model.py`, `artifacts/README.md`.
Optional reproducibility tools: `tools/build_kaggle_upload.py` and `tools/kaggle/`.
These optional notebook tools retain the earlier experiment protocol, not v1.

Keep the staged removal/moves of obsolete root model files. Do not also retain a
second copy of those implementations at the root. The private engine includes
its own legacy data/evaluation/service utilities because its offline CLI and
portable model format depend on them; these are not the team's public layers.

Exclude the earlier local integration prototypes (`src/data`, `src/evaluation`,
`src/service`, demo/benchmark scripts and their integration tests) from this
models-only commit. They still target the previous contracts. Do not stage the
whole repository with `git add -A`. Do not include datasets, trained weights,
model ZIPs, environments, caches or generated outputs.

The index may still contain the earlier adapters. Re-add the updated files before
committing, while retaining the already staged obsolete-file removals:

```powershell
Set-Location 'E:\project_OOP\game-recsys-oop'
git add -- src/models src/contracts src/__init__.py src/__main__.py src/cli.py
git add -- tests/test_team_models.py tests/test_numerical_core.py
git add -- .gitignore README.md pyproject.toml configs/model_config.json docs
git diff --cached --stat
git diff --cached --check
```

Add the optional importer only if wanted:
`git add -- scripts/install_model.py artifacts/README.md`.
Review shared `src/contracts` changes with the team because every layer imports
these types. No commit or push was performed as part of this local preparation.

## Public model API

Import models from `src.models`, and DTOs/errors from `src.contracts`.

```python
from src.contracts import ModelConfig
from src.models import BPRMatrixFactorization, PopularityRecommender, TextBPRMatrixFactorization

model = BPRMatrixFactorization(ModelConfig(n_factors=32, n_epochs=30, patience=5))
# train and validation below come from the Data teammate's DatasetSplits.
# evaluator must implement the new BaseEvaluator interface.
model.fit(train, validation_score=lambda candidate:
          evaluator.evaluate(candidate, train, validation).metrics["NDCG@10"])
results = model.recommend(user_id=0, train_dataset=train, top_k=10)
```

`fit(train_dataset, *, validation_score=None, item_features=None, verbose=False)`
only accepts `split_name="train"`. It converts shared mapped `Interaction` records
into positive and all-observed pairs. Negative reviews block BPR negative sampling
and are filtered from recommendations by default. Rows must already be cleaned,
deduplicated, binary and within the frozen mapping. No positives or no eligible
BPR triples gives a clear error.

The callback receives the public candidate with current epoch weights, dimensions,
and matching mapping/train fingerprints. Return a finite real scalar to maximize.
The best epoch is restored before fit completes. A failed refit, training step or
callback invalidates both adapter and backend; old parameters cannot still be
served as a successful fit. `best_epoch` and `history` expose training progress;
public history calls the callback value `validation_score`.

`score_items(user_id, item_ids)` preserves candidate order/duplicates, performs no
history filtering, and checks finite results. Empty candidates return `[]`.
`recommend` checks train/mapping fingerprints by content, filters all seen items,
and breaks score ties by ascending internal item ID. `config()` uses the common
`n_factors`, `n_epochs`, `random_seed`, `learning_rate`, `reg`, `patience`, and
`text_reg` (TextBPR only) names and is JSON serializable.

Popularity counts distinct positive train users per item. Its `config()` is `{}`;
it has no training hyperparameters or early-stopping callback. `score_global`
provides genuine global scores for a Service-controlled fallback. Model scoring
itself never substitutes user 0 or silently enables fallback.

## Train-only text features

`TextBPR.fit(..., item_features=features)` requires the shared `ItemFeatures` DTO,
not an unlabelled array. Raw item IDs must follow `train.item_mapper.to_raw(i)`
order. Mapping/train/text fingerprints must equal train's `mapping_fingerprint`,
`fingerprint`, and `review_fingerprint`. Values must be finite unit or zero vectors.
Metadata is a provenance assertion, not permission to relabel features created
from a different split. Pin the pretrained encoder to a fixed revision.

`src.models.text_features.build_item_features` creates this DTO and a report:

```python
from src.models.text_features import build_item_features

features, feature_report = build_item_features(
    train, encode_batch=frozen_encoder.encode,
    encoder_name=encoder_repository, encoder_revision=pinned_commit,
    batch_size=32,
)
model = TextBPRMatrixFactorization(ModelConfig(n_factors=32, n_epochs=30))
model.fit(train, item_features=features, validation_score=validation_callback)
```

The caller configures the fixed encoder for inference, including the correct
pooling/prompt and device. `encode_batch` must not learn from the supplied text.
The builder reads only `train.iter_reviews()`, checks each review's pair/label/time
against train, rejects duplicates or missing text, and includes positive AND
negative reviews. It averages normalized review embeddings by item, then normalizes
item vectors. Missing-text items remain zero. Save the returned feature report
alongside X; it records encoder identity, fingerprints, reviews used, items with
text and zero vectors. The Data report remains the source of missing/removed
review counts. Encoding remains separate from BPR training; CPU public adapters
do not automatically distribute training across GPUs.

## What Data must supply before real v1 training

- Read-only mapped `Interaction` records, mappers, dimensions and all three fingerprints.
- Train/validation/test with frozen mappings from ALL train interactions, including negative reviews.
- The v1 per-user 80/10/10 protocol (seed 42), dedup and unknown-positive policy.
- Real mapped review records from `iter_reviews()`, with label/time belonging to the retained interaction.
- `reviews.csv`, `train_reviews.jsonl`, `validation_reviews.jsonl`, `test_reviews.jsonl`, mappings and preparation report/checksums, as specified by the contract.

Evaluation supplies the v1 evaluator and NDCG@10 validation callback, then test
metrics only after model selection. Service supplies raw-ID lookup, metadata
filters, cold-start policy and the application API. The Model branch does not
implement those teammates' responsibilities.

## Existing Kaggle weights and notebooks

The private numerical engine and compact CPU format 3 are unchanged. The earlier
Kaggle notebook/CLI used an earlier preparation/mapping protocol (including the
70/15/15 experiment). Its weights can still be loaded through the legacy offline
path, but are NOT evidence of an experiment under the new v1 protocol. Do not fake
new fingerprints or claim those weights were trained on the new splits.

Retrain a fresh v1 experiment after the Data handoff is ready and the notebook
uses the new splits/features and public validation interface. Porting the legacy
GPU launcher/export to the v1 experiment protocol is a separate integration step;
changing a notebook split percentage alone does not complete it.

## Verification before push

From the repository root:

```powershell
python -m unittest discover -s tests -p test_team_models.py -v
python -m unittest discover -s tests -p test_numerical_core.py -v
python -m src --help
```

These tests use tiny contract-shaped fixtures, not a replacement Data layer or
production reviews. They cover gradients, numerical stability, callbacks/best
checkpoint, failure invalidation, fingerprints, negative-history exclusion,
feature provenance and common error boundaries. Full cross-team acceptance needs
the real v1 Data/Evaluation/Service implementations.
