"""Pretrained service facade and runtime backend for production / UI inference.

Supports:
1. Native PretrainedRecommender bundles (Contracts v1.1 CPU inference without retrain).
2. Adapter mode for BaseRecommender models and custom metadata mappings.
"""

from __future__ import annotations

from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set
import numpy as np

from src.contracts.dto import GameItem, Interaction, RecommendationResult, ScoredItem
from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    DataValidationError,
    InvalidParameterError,
    ModelNotFittedError,
    RecSysError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import (
    BaseRecommender,
    IdMapper,
    checked_indices,
    checked_scores,
)
from src.service.filters import FilterCriteria, game_passes_filters, parse_filter_criteria
from src.service.formatters import format_recommendations_table
from src.service.mappers import to_recommendation_result
from src.service.validators import validate_fallback_model, validate_id, validate_top_k


class SimpleIdMapper:
    """Minimal IdMapper implementation for wrapping sequence of raw IDs."""

    def __init__(self, raw_ids: Sequence[str]) -> None:
        self._raw_ids = tuple(str(r) for r in raw_ids)
        self._raw_to_idx = {r: i for i, r in enumerate(self._raw_ids)}
        self.size = len(self._raw_ids)

    def contains_raw(self, raw_id: str) -> bool:
        return raw_id in self._raw_to_idx

    def to_idx(self, raw_id: str) -> int:
        return self._raw_to_idx[raw_id]

    def to_raw(self, index: int) -> str:
        return self._raw_ids[index]


class PretrainedDatasetProxy:
    """Read-only InteractionDataset proxy backed by pre-computed metadata bundles."""

    def __init__(
        self,
        user_mapper: IdMapper,
        item_mapper: IdMapper,
        catalog: Dict[str, GameItem],
        user_seen_items: Dict[int, Set[int]],
        user_positive_items: Optional[Dict[int, Set[int]]] = None,
        mapping_fingerprint: str = "mfp-pretrained",
        fingerprint: str = "fp-pretrained",
        review_fingerprint: str = "rfp-pretrained",
    ) -> None:
        self.split_name = "train"
        self.user_mapper = user_mapper
        self.item_mapper = item_mapper
        self.num_users = user_mapper.size
        self.num_items = item_mapper.size
        self.mapping_fingerprint = mapping_fingerprint
        self.fingerprint = fingerprint
        self.review_fingerprint = review_fingerprint

        self._catalog = dict(catalog)
        self._user_seen = {u: frozenset(items) for u, items in user_seen_items.items()}
        self._user_pos = (
            {u: frozenset(items) for u, items in user_positive_items.items()}
            if user_positive_items is not None
            else self._user_seen
        )

        # Build minimal synthetic interactions sequence to fulfill protocol
        interactions_list: List[Interaction] = []
        for u, seen in self._user_seen.items():
            pos = self._user_pos.get(u, frozenset())
            for i in seen:
                rating = 1.0 if i in pos else 0.0
                interactions_list.append(Interaction(user_id=u, item_id=i, rating=rating, timestamp=0))
        self.interactions: Sequence[Interaction] = tuple(interactions_list)
        self.num_interactions = len(self.interactions)

    def get_game_by_raw_id(self, raw_game_id: str) -> GameItem:
        if raw_game_id not in self._catalog:
            raise UnknownItemError(f"Game '{raw_game_id}' is not in catalog.")
        return self._catalog[raw_game_id]

    def get_user_seen_items(self, user_id: int) -> FrozenSet[int]:
        return self._user_seen.get(user_id, frozenset())

    def get_user_positive_items(self, user_id: int) -> FrozenSet[int]:
        return self._user_pos.get(user_id, frozenset())

    def iter_reviews(self) -> Iterable[Any]:
        return iter(())


def _is_pretrained_recommender(obj: Any) -> bool:
    if obj is None:
        return False
    if obj.__class__.__name__ == "PretrainedRecommender":
        return True
    return (
        hasattr(obj, "recommend_games")
        and hasattr(obj, "get_user_played_games")
        and hasattr(obj, "get_game_details")
        and not isinstance(obj, BaseRecommender)
    )


class PretrainedServiceFacade:
    """Production and UI facade coordinating inference from pre-trained model bundles.

    Supports both:
    1. Real PretrainedRecommender instances from src.models.pretrained (Contracts v1.1).
    2. BaseRecommender instances with explicit mapper/catalog metadata (Adapter mode).
    """

    def __init__(
        self,
        model: Any,
        user_mapper: Optional[IdMapper] = None,
        item_mapper: Optional[IdMapper] = None,
        catalog: Optional[Dict[str, GameItem]] = None,
        user_seen_items: Optional[Dict[int, Set[int]]] = None,
        user_positive_items: Optional[Dict[int, Set[int]]] = None,
        *,
        fallback_model: Optional[BaseRecommender] = None,
        allow_cold_start_fallback: bool = False,
        mapping_fingerprint: str = "mfp-pretrained",
        train_fingerprint: str = "fp-pretrained",
        catalog_lookup: Optional[Callable[[str], GameItem]] = None,
    ) -> None:
        if isinstance(model, (str, Path)):
            facade = self.load(
                model,
                allow_cold_start_fallback=allow_cold_start_fallback,
                catalog_lookup=catalog_lookup,
            )
            self._is_native = True
            self._recommender = facade._recommender
            self._model = facade._recommender
            self._allow_cold_start_fallback = allow_cold_start_fallback
            self._extra_seen_raw: Dict[str, Set[str]] = defaultdict(set)
            self._dataset_proxy: Optional[PretrainedDatasetProxy] = None
            self._fallback_model = getattr(self._recommender, "_popularity", None)
            return

        is_native = _is_pretrained_recommender(model) or (
            user_mapper is None and hasattr(model, "recommend_games") and hasattr(model, "get_game_details")
        )

        if is_native:
            self._is_native = True
            self._recommender = model
            self._model = model
            self._allow_cold_start_fallback = allow_cold_start_fallback
            if hasattr(model, "allow_cold_start_fallback"):
                model.allow_cold_start_fallback = allow_cold_start_fallback
            self._extra_seen_raw = defaultdict(set)
            self._dataset_proxy = None
            self._fallback_model = getattr(model, "_popularity", None)
        else:
            if not isinstance(model, BaseRecommender):
                raise InvalidParameterError("model must be an instance of BaseRecommender or PretrainedRecommender.")
            if not bool(getattr(model, "is_fitted", False)):
                raise ModelNotFittedError(f"Model '{model.name}' must be fitted before initialization.")
            if user_mapper is None or item_mapper is None or catalog is None or user_seen_items is None:
                raise InvalidParameterError(
                    "user_mapper, item_mapper, catalog, and user_seen_items are required for BaseRecommender adapter mode."
                )

            self._is_native = False
            self._recommender = None
            self._model = model
            self._fallback_model = fallback_model
            self._allow_cold_start_fallback = allow_cold_start_fallback
            self._user_mapper = user_mapper
            self._item_mapper = item_mapper
            self._catalog = dict(catalog)
            self._user_seen_items = {u: set(items) for u, items in user_seen_items.items()}
            self._user_positive_items = (
                {u: set(items) for u, items in user_positive_items.items()}
                if user_positive_items is not None
                else {u: set(items) for u, items in user_seen_items.items()}
            )
            self._extra_seen: Dict[int, Set[int]] = defaultdict(set)
            self._extra_seen_raw = defaultdict(set)

            self._dataset_proxy = PretrainedDatasetProxy(
                user_mapper=user_mapper,
                item_mapper=item_mapper,
                catalog=self._catalog,
                user_seen_items=self._user_seen_items,
                user_positive_items=self._user_positive_items,
                mapping_fingerprint=mapping_fingerprint,
                fingerprint=train_fingerprint,
            )

            if fallback_model is not None:
                validate_fallback_model(fallback_model)
                if not bool(getattr(fallback_model, "is_fitted", False)):
                    fallback_model.fit(self._dataset_proxy)

    @classmethod
    def load(
        cls,
        folder: Any,
        *,
        allow_cold_start_fallback: bool = False,
        catalog_lookup: Optional[Callable[[str], GameItem]] = None,
    ) -> "PretrainedServiceFacade":
        """Load a trained model bundle and return a PretrainedServiceFacade."""
        for module_name in (
            "src.models.pretrained",
            "src.models",
            "models.pretrained",
            "models",
        ):
            try:
                mod = import_module(module_name)
                rec_cls = getattr(mod, "PretrainedRecommender", None)
                if rec_cls is not None and hasattr(rec_cls, "load"):
                    recommender = rec_cls.load(
                        folder,
                        allow_cold_start_fallback=allow_cold_start_fallback,
                        catalog_lookup=catalog_lookup,
                    )
                    return cls(recommender, allow_cold_start_fallback=allow_cold_start_fallback)
            except (ImportError, AttributeError):
                continue
        raise ConfigurationError(
            "PretrainedRecommender is not available. Ensure src.models.pretrained is importable."
        )

    @classmethod
    def from_pretrained(
        cls,
        folder: Any,
        *,
        allow_cold_start_fallback: bool = False,
        catalog_lookup: Optional[Callable[[str], GameItem]] = None,
    ) -> "PretrainedServiceFacade":
        """Alias for PretrainedServiceFacade.load."""
        return cls.load(
            folder,
            allow_cold_start_fallback=allow_cold_start_fallback,
            catalog_lookup=catalog_lookup,
        )

    @property
    def model(self) -> Any:
        return self._recommender if self._is_native else self._model

    @property
    def model_name(self) -> str:
        if self._is_native:
            return getattr(self._recommender, "model_name", self._recommender.__class__.__name__)
        return self._model.name

    @property
    def fallback_model(self) -> Optional[Any]:
        if self._is_native:
            return getattr(self._recommender, "_popularity", getattr(self, "_fallback_model", None))
        return self._fallback_model

    @property
    def allow_cold_start_fallback(self) -> bool:
        if self._is_native and hasattr(self._recommender, "allow_cold_start_fallback"):
            return bool(self._recommender.allow_cold_start_fallback)
        return self._allow_cold_start_fallback

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be boolean.")
        self._allow_cold_start_fallback = value
        if self._is_native and hasattr(self._recommender, "allow_cold_start_fallback"):
            self._recommender.allow_cold_start_fallback = value

    @property
    def provenance(self) -> dict:
        if self._is_native:
            return getattr(self._recommender, "provenance", {})
        return {
            "protocol": "service-adapter-v1",
            "mapping_fingerprint": self.dataset.mapping_fingerprint,
            "train_fingerprint": self.dataset.fingerprint,
        }

    @property
    def dataset(self) -> PretrainedDatasetProxy:
        if self._dataset_proxy is not None:
            return self._dataset_proxy
        if self._is_native:
            self._dataset_proxy = self._build_dataset_proxy_from_recommender(self._recommender)
            return self._dataset_proxy
        raise ConfigurationError("Dataset proxy is not initialized.")

    @classmethod
    def _build_dataset_proxy_from_recommender(cls, recommender: Any) -> PretrainedDatasetProxy:
        user_ids = tuple(getattr(recommender, "_users", {}).keys())
        item_ids = tuple(getattr(recommender, "_items", ()))
        user_mapper = SimpleIdMapper(user_ids)
        item_mapper = SimpleIdMapper(item_ids)

        catalog: Dict[str, GameItem] = {}
        for gid in item_ids:
            try:
                catalog[gid] = recommender.get_game_details(gid)
            except Exception:
                catalog[gid] = GameItem(item_id=gid, title=gid)

        user_seen: Dict[int, Set[int]] = {}
        if hasattr(recommender, "_history"):
            for u_idx in range(len(user_ids)):
                try:
                    hist = recommender._history(u_idx)
                    if len(hist) > 0:
                        user_seen[u_idx] = set(int(i) for i in hist)
                except Exception:
                    pass

        provenance = getattr(recommender, "provenance", {}) or {}
        fp = provenance.get("data_fingerprint", "fp-pretrained")

        return PretrainedDatasetProxy(
            user_mapper=user_mapper,
            item_mapper=item_mapper,
            catalog=catalog,
            user_seen_items=user_seen,
            mapping_fingerprint=fp,
            fingerprint=fp,
            review_fingerprint="rfp-pretrained",
        )

    def record_user_interaction(
        self,
        raw_user_id: str,
        raw_game_id: str,
        rating: float = 1.0,
    ) -> None:
        """Record dynamic runtime interaction (extra_seen) from session/UI."""
        raw_user_id = validate_id(raw_user_id, "raw_user_id")
        raw_game_id = validate_id(raw_game_id, "raw_game_id")

        if self._is_native:
            if hasattr(self._recommender, "_users") and raw_user_id not in self._recommender._users:
                raise UnknownUserError(f"User '{raw_user_id}' is not in user mapper.")
            if hasattr(self._recommender, "_items") and raw_game_id not in getattr(self._recommender, "_item_map", set(self._recommender._items)):
                raise UnknownItemError(f"Game '{raw_game_id}' is not in item mapper.")
            self._extra_seen_raw[raw_user_id].add(raw_game_id)
        else:
            if not self._user_mapper.contains_raw(raw_user_id):
                raise UnknownUserError(f"User '{raw_user_id}' is not in user mapper.")
            if not self._item_mapper.contains_raw(raw_game_id):
                raise UnknownItemError(f"Game '{raw_game_id}' is not in item mapper.")
            u_idx = int(self._user_mapper.to_idx(raw_user_id))
            i_idx = int(self._item_mapper.to_idx(raw_game_id))
            self._extra_seen[u_idx].add(i_idx)
            self._extra_seen_raw[raw_user_id].add(raw_game_id)
            if rating == 1.0:
                self._user_positive_items.setdefault(u_idx, set()).add(i_idx)

    def _get_combined_seen(self, user_idx: int) -> Set[int]:
        seen = set(self._user_seen_items.get(user_idx, ()))
        seen.update(self._extra_seen.get(user_idx, ()))
        return seen

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
        *,
        extra_seen: Any = (),
    ) -> List[RecommendationResult]:
        raw_user_id = validate_id(raw_user_id, "raw_user_id")
        top_k = validate_top_k(top_k)
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be boolean.")

        if self._is_native:
            combined_extra: Set[str] = set()
            if extra_seen:
                if isinstance(extra_seen, (str, bytes)):
                    raise InvalidParameterError("extra_seen must be a collection of original game IDs")
                try:
                    combined_extra.update(str(g).strip() for g in extra_seen if str(g).strip())
                except TypeError as exc:
                    raise InvalidParameterError("extra_seen must be a collection of original game IDs") from exc
            if raw_user_id in self._extra_seen_raw:
                combined_extra.update(self._extra_seen_raw[raw_user_id])

            return self._recommender.recommend_games(
                raw_user_id,
                top_k=top_k,
                exclude_played=exclude_played,
                genre_filter=genre_filter,
                max_price=max_price,
                extra_seen=tuple(combined_extra),
            )

        criteria = parse_filter_criteria(genre_filter, max_price)
        is_fallback = False
        mapped_user_idx: Optional[int] = None

        if self._user_mapper.contains_raw(raw_user_id):
            mapped_user_idx = int(self._user_mapper.to_idx(raw_user_id))
            user_pos = self._user_positive_items.get(mapped_user_idx, set())
            if not user_pos:
                if not self._allow_cold_start_fallback:
                    raise ColdStartError(
                        f"User '{raw_user_id}' has no positive train interactions "
                        "and cold-start fallback is disabled."
                    )
                is_fallback = True
        else:
            if not self._allow_cold_start_fallback:
                raise UnknownUserError(f"User '{raw_user_id}' is not in mapping.")
            is_fallback = True

        candidates = np.arange(self._item_mapper.size, dtype=np.int64)
        if exclude_played and mapped_user_idx is not None:
            seen_items = self._get_combined_seen(mapped_user_idx)
            if seen_items:
                seen_arr = checked_indices(list(seen_items), self._item_mapper.size)
                candidates = np.setdiff1d(candidates, seen_arr, assume_unique=True)

        if candidates.size == 0:
            return []

        active_model_name = self.model_name
        if is_fallback:
            if self._fallback_model is None:
                raise ConfigurationError("Cold-start fallback triggered but fallback_model is not configured.")
            raw_scores = self._fallback_model.score_global(candidates)
            scores = checked_scores(raw_scores, len(candidates))
            active_model_name = getattr(self._fallback_model, "name", "Popularity")
        else:
            assert mapped_user_idx is not None
            raw_scores = self._model.score_items(mapped_user_idx, candidates)
            scores = checked_scores(raw_scores, len(candidates))

        order = np.lexsort((candidates, -scores))

        results: List[RecommendationResult] = []
        for idx in order:
            item_idx = int(candidates[idx])
            raw_game_id = str(self._item_mapper.to_raw(item_idx))
            if raw_game_id not in self._catalog:
                continue
            game = self._catalog[raw_game_id]
            if not game_passes_filters(game, criteria):
                continue
            scored = ScoredItem(item_idx=item_idx, score=float(scores[idx]))
            results.append(to_recommendation_result(
                scored, game, len(results) + 1, active_model_name, is_fallback
            ))
            if len(results) == top_k:
                break

        return results

    def get_user_played_games(self, raw_user_id: str) -> List[GameItem]:
        raw_user_id = validate_id(raw_user_id, "raw_user_id")
        if self._is_native:
            games = list(self._recommender.get_user_played_games(raw_user_id))
            if raw_user_id in self._extra_seen_raw:
                seen_ids = {g.item_id for g in games}
                for gid in sorted(self._extra_seen_raw[raw_user_id]):
                    if gid not in seen_ids:
                        try:
                            games.append(self.get_game_details(gid))
                            seen_ids.add(gid)
                        except UnknownItemError:
                            continue
            return games

        if not self._user_mapper.contains_raw(raw_user_id):
            raise UnknownUserError(f"User '{raw_user_id}' is not in mapping.")
        u_idx = int(self._user_mapper.to_idx(raw_user_id))
        seen_indices = sorted(self._get_combined_seen(u_idx))
        games = []
        for i_idx in seen_indices:
            raw_id = str(self._item_mapper.to_raw(i_idx))
            if raw_id not in self._catalog:
                raise UnknownItemError(f"Game '{raw_id}' is not in catalog.")
            games.append(self._catalog[raw_id])
        return games

    def get_game_details(self, raw_game_id: str) -> GameItem:
        raw_game_id = validate_id(raw_game_id, "raw_game_id")
        if self._is_native:
            return self._recommender.get_game_details(raw_game_id)
        if raw_game_id not in self._catalog:
            raise UnknownItemError(f"Game '{raw_game_id}' is not in catalog.")
        return self._catalog[raw_game_id]

    def get_catalog_metadata(self, raw_game_id: str) -> dict:
        raw_game_id = validate_id(raw_game_id, "raw_game_id")
        if self._is_native and hasattr(self._recommender, "get_catalog_metadata"):
            return self._recommender.get_catalog_metadata(raw_game_id)
        game = self.get_game_details(raw_game_id)
        return {
            "title": game.title,
            "genres": list(game.genres),
            "price": game.price,
            "developer": game.developer,
        }

    def get_available_user_ids(self, limit: Optional[int] = None) -> tuple[str, ...]:
        """Return raw user IDs available in the pretrained mapping."""
        if self._is_native and hasattr(self._recommender, "_users"):
            raw_ids = tuple(self._recommender._users.keys())
        else:
            raw_ids = tuple(self.dataset.user_mapper.to_raw(i) for i in range(self.dataset.num_users))
        if limit is not None and limit > 0 and len(raw_ids) > limit:
            step = max(1, len(raw_ids) // limit)
            return raw_ids[::step][:limit]
        return raw_ids

    def get_available_game_ids(self) -> tuple[str, ...]:
        """Return raw game IDs available in the pretrained catalog mapping."""
        if self._is_native and hasattr(self._recommender, "_items"):
            return tuple(self._recommender._items)
        mapper = self.dataset.item_mapper
        return tuple(str(mapper.to_raw(i)) for i in range(mapper.size))

    def get_available_genres(self) -> tuple[str, ...]:
        """Return sorted catalog genres for UI filters."""
        genres: Set[str] = set()
        for gid in self.get_available_game_ids():
            try:
                game = self.get_game_details(gid)
                genres.update(str(g).strip() for g in game.genres if str(g).strip())
            except Exception:
                continue
        return tuple(sorted(genres, key=str.casefold))

    def recommend_baseline(
        self,
        raw_user_id: Optional[str] = None,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
    ) -> List[RecommendationResult]:
        """Generate baseline recommendations using popularity fallback."""
        if self._is_native:
            prev = getattr(self._recommender, "allow_cold_start_fallback", False)
            try:
                self._recommender.allow_cold_start_fallback = True
                bogus_user = "__baseline_popularity_sentinel__"
                extra = set(self._extra_seen_raw.get(raw_user_id, ())) if raw_user_id else set()
                if raw_user_id and exclude_played and hasattr(self._recommender, "_users") and hasattr(self._recommender, "_history"):
                    if raw_user_id in self._recommender._users:
                        u_idx = self._recommender._users[raw_user_id]
                        for i_idx in self._recommender._history(u_idx):
                            extra.add(self._recommender._items[i_idx])
                return self._recommender.recommend_games(
                    bogus_user,
                    top_k=top_k,
                    exclude_played=exclude_played,
                    genre_filter=genre_filter,
                    max_price=max_price,
                    extra_seen=tuple(extra),
                )
            finally:
                self._recommender.allow_cold_start_fallback = prev
        else:
            if self._fallback_model is None:
                raise ConfigurationError("No fallback model configured for baseline.")
            return self.recommend_games(
                raw_user_id or "__sentinel__",
                top_k=top_k,
                exclude_played=exclude_played,
                genre_filter=genre_filter,
                max_price=max_price,
            )

    @staticmethod
    def format_recommendations_table(results: Sequence[RecommendationResult]) -> str:
        return format_recommendations_table(results)


__all__ = [
    "SimpleIdMapper",
    "PretrainedDatasetProxy",
    "PretrainedServiceFacade",
]
