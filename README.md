# Recommendation models — Member 2

This branch represents the model folder. Model files are placed directly at the branch root, with no `src/` or nested `models/` directory.

```text
__init__.py              # Public exports when integrated as a package
base.py                  # BaseRecommender: fit/score interface and fitted checks
_utils.py                # Dimensions, mapped-pair validation and pair sets
popularity.py            # PopularityRecommender
matrix_factorization.py  # BPRMatrixFactorization / MatrixFactorizationRecommender
text_bpr.py              # TextBPRMatrixFactorization
models.py                # Compatibility exports for the former monolithic module
requirements.txt         # NumPy dependency for training and scoring
text_features.py         # Existing offline encoder/feature pipeline (integration helper)
cli.py                   # Existing full-pipeline CLI (integration helper)
tests/
    test_models.py       # Model-only gradient, validation and training tests
```

## Your model files

- `popularity.py` counts distinct training users per item; repeated interactions count once.
- `matrix_factorization.py` contains the BPR objective, analytical gradients, factor initialization, negative sampling, SGD and validation-based early stopping.
- `text_bpr.py` adds frozen item features `X` and a learned projection `W`, with score `P[u] @ (Q[i] + W @ X[i])`.
- `base.py` holds the current array-based abstract model interface and fitted/index checks.
- `_utils.py` holds shared model validation helpers.

`MatrixFactorizationRecommender` is an alias for `BPRMatrixFactorization`, matching the class name in the team diagram. It is the existing implicit-feedback BPR algorithm trained with SGD. The runtime class name stays `BPRMatrixFactorization` to preserve saved model metadata. All existing training and scoring definitions are unchanged by the split.

## Run the models on this branch

Use Python 3.10+ and run from the checkout root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```python
import numpy as np
from popularity import PopularityRecommender
from matrix_factorization import MatrixFactorizationRecommender
from text_bpr import TextBPRMatrixFactorization

train = np.array([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=np.int64)
n_users, n_items = 2, 4

popularity = PopularityRecommender().fit(train, n_users, n_items)
model = MatrixFactorizationRecommender(factors=8, epochs=10, seed=42)
model.fit(train, n_users, n_items, observed=train, verbose=False)
scores = model.score_all(user_idx=0)
assert scores.shape == (n_items,)

# Synthetic features only for this example. Real features must align to the
# same training item mapping and use training review text only.
features = np.eye(n_items)
text_model = TextBPRMatrixFactorization(factors=8, epochs=10, seed=42)
text_model.fit(train, n_users, n_items, item_features=features, verbose=False)
```

The compatibility module supports `from models import BPRMatrixFactorization`. The new individual model modules also support relative imports when this folder is integrated into a Python package.

## Input/output contract for teammates

- `interactions`: a nonempty integer array of shape `(N, 2)` containing positive training `(user_idx, item_idx)` pairs. The model deduplicates pairs.
- `n_users`, `n_items`: positive integer dimensions; mapped IDs must be in range.
- `observed` (BPR/TextBPR): optional mapped training history, including low ratings, excluded from sampled negatives. Do not include validation/test history.
- `item_features` (TextBPR): finite values with shape `(n_items, embedding_dimension)`; rows must have unit norm or be zero for missing text.
- `validation_score` (BPR/TextBPR): an optional callback receiving the current fitted model and returning a finite score to maximize. The evaluator supplies the callback.
- `score_all(user_idx)`: a score vector of shape `(n_items,)`.
- `score(user_idx, item_idx)`: a scalar score. Fit the model before scoring.

File loading, cleaning, ID mapping and splitting belong to the data layer. Metrics belong to evaluation. Top-K filtering, original IDs, unknown-user fallback and checkpoint bundles belong to the service layer.

The inherited `cli.py` and `text_features.py` are retained unchanged apart from their location. They depend on the data, evaluation and service modules from the full project, which are not included on this model-only branch; they are not standalone entry points here. The core model modules and tests need only NumPy. Encoding new review text additionally requires the original PyTorch/Transformers environment and local MiniLM weights after pipeline integration.

The team's shared `contracts/` and `InteractionDataset` implementation are not present on this branch. `base.py` preserves the current array-based API. Adapting to a shared `BaseRecommender.fit(dataset)` contract should happen once that interface is agreed; this file split does not change the public training API.
