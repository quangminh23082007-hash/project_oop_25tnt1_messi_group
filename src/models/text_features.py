"""Build the official v1 item matrix from train reviews and a fixed encoder.

The caller supplies encode_batch(texts); it must perform inference only using
the declared pretrained encoder revision. This module never reads another split.
"""
from numbers import Integral
import numpy as np
from src.contracts.dto import ItemFeatures, ReviewRecord
from src.contracts.exceptions import ConfigurationError, DataValidationError, EmptyDatasetError
from ._adapter import training_pairs


def build_item_features(train_dataset, encode_batch, *, encoder_name, encoder_revision,
                        batch_size=32):
    """Return (ItemFeatures, report), averaging normalized train-review vectors.

    Both positive and negative reviews contribute. Missing item text produces a
    zero vector. Caller must persist the returned report beside the features.
    """
    training_pairs(train_dataset)
    if not callable(encode_batch):
        raise ConfigurationError("encode_batch must be a frozen-encoder inference callable")
    if any(not isinstance(v, str) or not v.strip() for v in (encoder_name, encoder_revision)):
        raise ConfigurationError("Declare the fixed pretrained encoder name and revision")
    if encoder_revision.strip().lower() in {"main", "master", "latest"}:
        raise ConfigurationError("Pin a model commit or immutable revision, not a moving branch")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ConfigurationError("batch_size must be a positive integer")
    records = {(r.user_id, r.item_id): (r.rating, r.timestamp) for r in train_dataset.interactions}
    used_pairs = set()
    counts = np.zeros(train_dataset.num_items, dtype=np.int64)
    sums = None

    def consume(batch):
        nonlocal sums
        try:
            vectors = np.asarray(encode_batch([r.text for r in batch]))
        except Exception as exc:
            raise DataValidationError(f"Encoder failed: {exc}") from exc
        if (vectors.dtype.kind not in "fiu" or vectors.ndim != 2
                or vectors.shape[0] != len(batch) or vectors.shape[1] == 0
                or not np.isfinite(vectors).all()):
            raise DataValidationError("Encoder must return a finite real (batch, dimension) matrix")
        vectors = vectors.astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise DataValidationError("Encoder returned an invalid/zero review vector")
        vectors /= norms
        if sums is None:
            sums = np.zeros((train_dataset.num_items, vectors.shape[1]), dtype=np.float64)
        if vectors.shape[1] != sums.shape[1]:
            raise DataValidationError("Encoder dimension changed between batches")
        ids = [r.item_id for r in batch]
        np.add.at(sums, ids, vectors)
        np.add.at(counts, ids, 1)

    batch = []
    for review in train_dataset.iter_reviews():
        if not isinstance(review, ReviewRecord):
            raise DataValidationError("iter_reviews must yield shared ReviewRecord instances")
        pair = (review.user_id, review.item_id)
        if records.get(pair) != (review.rating, review.timestamp):
            raise DataValidationError("Review pair/label/timestamp does not match the train record")
        if pair in used_pairs:
            raise DataValidationError("Duplicate review for a deduplicated train interaction")
        used_pairs.add(pair)
        batch.append(review)
        if len(batch) == batch_size:
            consume(batch)
            batch.clear()
    if batch:
        consume(batch)
    if sums is None:
        raise EmptyDatasetError("No nonblank train reviews; TextBPR cannot be prepared")
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    # Exact cancellation also yields zero; the report distinguishes this from missing text.
    np.divide(sums, norms, out=sums, where=norms > 0)
    features = ItemFeatures(
        sums, tuple(train_dataset.item_mapper.to_raw(i) for i in range(train_dataset.num_items)),
        train_dataset.mapping_fingerprint, train_dataset.fingerprint,
        train_dataset.review_fingerprint, f"{encoder_name}@{encoder_revision}")
    report = dict(reviews_used=int(counts.sum()), items_with_text=int((counts > 0).sum()),
                  zero_vectors=int((norms[:, 0] == 0).sum()),
                  items_without_text=int((counts == 0).sum()),
                  encoder_name=features.encoder_name, aggregation="mean_normalized_reviews_then_l2",
                  mapping_fingerprint=features.mapping_fingerprint,
                  train_fingerprint=features.train_fingerprint, text_fingerprint=features.text_fingerprint)
    return features, report
