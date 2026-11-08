"""Translate Contracts v1 datasets to the preserved array engine."""
from dataclasses import replace
from math import isfinite
from numbers import Integral, Real
import numpy as np
from src.contracts.dto import Interaction, ModelConfig
from src.contracts.exceptions import (
    ConfigurationError, DataValidationError, EmptyDatasetError, TrainingError,
)
from src.contracts.interfaces import checked_indices, checked_scores
from .base import BaseRecommender


def engine_options(config, *, factors=None, learning_rate=None, reg=None,
                   epochs=None, seed=None, patience=None, text_reg=None):
    config = ModelConfig() if config is None else config
    if not isinstance(config, ModelConfig):
        raise ConfigurationError("config must be the shared ModelConfig")
    overrides = dict(n_factors=factors, learning_rate=learning_rate, reg=reg,
                     n_epochs=epochs, random_seed=seed, patience=patience, text_reg=text_reg)
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})
    return dict(factors=int(config.n_factors), learning_rate=float(config.learning_rate),
                reg=float(config.reg), epochs=int(config.n_epochs), seed=int(config.random_seed),
                patience=int(config.patience), text_reg=float(config.text_reg))


def training_pairs(dataset):
    """Require mapped, deduplicated train records, including all negative history."""
    if dataset is None:
        raise EmptyDatasetError("A train dataset is required")
    if getattr(dataset, "split_name", None) != "train":
        raise DataValidationError("fit requires split_name='train'")
    for name in ("num_users", "num_items", "num_interactions"):
        value = getattr(dataset, name, None)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise DataValidationError(f"Invalid dataset {name}")
        if value == 0:
            raise EmptyDatasetError(f"Train dataset has zero {name}")
    for name in ("mapping_fingerprint", "fingerprint", "review_fingerprint"):
        value = getattr(dataset, name, None)
        if not isinstance(value, str) or not value.strip():
            raise DataValidationError(f"Dataset must declare {name}")
    for name, count in (("user_mapper", dataset.num_users), ("item_mapper", dataset.num_items)):
        if getattr(getattr(dataset, name, None), "size", None) != count:
            raise DataValidationError(f"{name}.size must match dataset dimensions")
    positive, observed = [], []
    try:
        for record in dataset.interactions:
            if not isinstance(record, Interaction):
                raise DataValidationError("interactions must contain shared, mapped Interaction records")
            if record.user_id >= dataset.num_users or record.item_id >= dataset.num_items:
                raise DataValidationError("Interaction IDs exceed the frozen mapping")
            pair = (record.user_id, record.item_id)
            observed.append(pair)
            if record.rating == 1.0:
                positive.append(pair)
    except (AttributeError, TypeError) as exc:
        raise DataValidationError("Dataset must expose mapped interactions") from exc
    if len(observed) != dataset.num_interactions:
        raise DataValidationError("num_interactions does not match the train records")
    observed = np.asarray(observed, dtype=np.int64).reshape(-1, 2)
    if len(np.unique(observed, axis=0)) != len(observed):
        raise DataValidationError("Data must deduplicate user-item pairs before fitting")
    if not positive:
        raise EmptyDatasetError("No positive train interactions")
    return np.asarray(positive, dtype=np.int64).reshape(-1, 2), observed


class ArrayModelAdapter(BaseRecommender):
    supports_callback = True

    @property
    def core_model(self):
        """Private numerical engine for legacy offline export tools."""
        return self._engine

    @property
    def history(self):
        return [{("validation_score" if k == "validation_ndcg" else k): v
                 for k, v in row.items()} for row in getattr(self._engine, "history", ())]

    @property
    def best_epoch(self):
        return getattr(self._engine, "best_epoch", None)

    def _invalidate(self):
        self._is_fitted = self._engine._is_fitted = False
        self._mapping_fingerprint = self._train_fingerprint = None
        self._num_users = self._num_items = 0

    def _feature_options(self, dataset, item_features):
        if item_features is not None:
            raise ConfigurationError(f"{self.name} does not accept item_features")
        return {}

    def fit(self, train_dataset, *, validation_score=None, item_features=None, verbose=False):
        self._invalidate()
        training_started = False
        try:
            if not isinstance(verbose, bool):
                raise ConfigurationError("verbose must be boolean")
            if validation_score is not None and (not self.supports_callback or not callable(validation_score)):
                raise ConfigurationError(f"{self.name} requires a supported callable validation_score")
            pairs, observed = training_pairs(train_dataset)
            options = self._feature_options(train_dataset, item_features)
            if self.supports_callback:
                counts = np.bincount(observed[:, 0], minlength=train_dataset.num_users)
                if not np.any(counts[pairs[:, 0]] < train_dataset.num_items):
                    raise EmptyDatasetError("No valid BPR triples: positive users have seen every item")
            self._num_users, self._num_items = int(train_dataset.num_users), int(train_dataset.num_items)
            self._mapping_fingerprint = train_dataset.mapping_fingerprint
            self._train_fingerprint = train_dataset.fingerprint

            def validate(_backend):
                # Evaluator receives this PUBLIC candidate, with current-epoch weights.
                self._is_fitted = True
                try:
                    value = validation_score(self)
                    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
                        raise ValueError("validation_score must return a finite real scalar to maximize")
                    return float(value)
                finally:
                    self._is_fitted = False

            training_started = True
            self._engine.fit(pairs, self.num_users, self.num_items, observed=observed,
                             validation_score=validate if validation_score is not None else None,
                             verbose=verbose, **options)
            # Backend has now restored the best checkpoint and cleared its score cache.
            self._is_fitted = True
            return self
        except BaseException as exc:
            self._invalidate()
            if training_started and isinstance(exc, Exception):
                raise TrainingError(f"{self.name} training/validation failed: {exc}") from exc
            raise

    def score_items(self, user_id, item_ids):
        self._check_fitted()
        self._check_user(user_id)
        indices = checked_indices(item_ids, self.num_items)
        if not indices.size:
            return []
        try:
            all_scores = checked_scores(self._engine.score_all(int(user_id)), self.num_items)
            return all_scores[indices].copy()
        except (TypeError, ValueError, RuntimeError) as exc:
            if isinstance(exc, DataValidationError):
                raise
            raise DataValidationError(f"{self.name} scoring failed: {exc}") from exc

    def config(self):
        if not self.supports_callback:
            return {}
        options = self._engine.config()
        names = {"factors": "n_factors", "epochs": "n_epochs", "seed": "random_seed"}
        return {names.get(k, k): v for k, v in options.items()}
