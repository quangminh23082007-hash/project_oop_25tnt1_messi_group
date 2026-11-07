"""Distinct positive-train-user popularity, including global fallback scoring."""
from src.contracts.exceptions import ConfigurationError
from src.contracts.interfaces import checked_indices, checked_scores
from ._adapter import ArrayModelAdapter
from ._array.popularity import PopularityRecommender as ArrayPopularity


class PopularityRecommender(ArrayModelAdapter):
    supports_callback = False

    def __init__(self, **unsupported):
        super().__init__("Popularity")
        if unsupported:
            raise ConfigurationError(f"Popularity accepts no hyperparameters: {', '.join(sorted(unsupported))}")
        self._engine = ArrayPopularity()

    def score_global(self, item_ids):
        self._check_fitted()
        indices = checked_indices(item_ids, self.num_items)
        if not indices.size:
            return []
        return checked_scores(self._engine.counts, self.num_items)[indices].copy()
