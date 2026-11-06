"""Inference-only entry point for the original Kaggle format-3 CPU bundles.

This does not fabricate a v1 training Dataset or claim v1 evaluation provenance.
Load once at backend startup. The large encoder and GPU are not used here.
"""
from numbers import Integral, Real
import math
import numpy as np
from src.contracts.dto import GameItem, RecommendationResult
from src.contracts.exceptions import (ConfigurationError, DataValidationError,
                                     InvalidParameterError, UnknownItemError, UnknownUserError)
from ._array.service import load_bundle
from ._array.evaluation import top_k as top_k_indices


class PretrainedRecommender:
    @classmethod
    def load(cls, folder, *, allow_cold_start_fallback=False, catalog_lookup=None):
        if not isinstance(allow_cold_start_fallback, bool):
            raise ConfigurationError("allow_cold_start_fallback must be boolean")
        if catalog_lookup is not None and not callable(catalog_lookup):
            raise ConfigurationError("catalog_lookup must return a shared GameItem by raw game ID")
        obj = cls()
        try:
            obj._model, obj._popularity, info, obj._seen = load_bundle(folder)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DataValidationError(f"Cannot load the trained model bundle: {exc}") from exc
        obj._users = {u: i for i, u in enumerate(info['user_ids'])}
        obj._items = tuple(info['item_ids'])
        obj._item_map = {item: i for i, item in enumerate(obj._items)}
        obj._catalog = cls._validate_catalog(info.get("catalog"))
        obj.model_name = obj._model.source
        obj.allow_cold_start_fallback = allow_cold_start_fallback
        obj._catalog_lookup = catalog_lookup
        obj.provenance = {'protocol': 'legacy-kaggle-format-3',
                          'data_fingerprint': info['data_fingerprint'],
                          'data_config': info.get('data_config', {}),
                          'feature_info': info.get('feature_info')}
        # load_bundle validates and sorts unique (user,item) pairs. Keep the array
        # instead of building over a million Python sets at application startup.
        return obj

    @staticmethod
    def _validate_catalog(catalog):
        if not isinstance(catalog, dict):
            raise DataValidationError(
                "catalog must be a JSON object containing game metadata"
            )

        for game_id, row in catalog.items():
            if not isinstance(game_id, str) or not game_id.strip():
                raise DataValidationError(
                    "catalog keys must be nonempty game ID strings"
                )

            if not isinstance(row, dict):
                raise DataValidationError(
                    f"catalog entry {game_id!r} must be a JSON object"
                )

            for field in ("title", "image_url"):
                value = row.get(field)
                if value is not None and not isinstance(value, str):
                    raise DataValidationError(
                        f"catalog[{game_id!r}].{field} must be a string or null"
                    )

            title = row.get("title")
            if isinstance(title, str) and title and not title.strip():
                raise DataValidationError(
                    f"catalog[{game_id!r}].title cannot contain only whitespace"
                )

            categories = row.get("categories", [])
            if not isinstance(categories, list) or any(
                not isinstance(category, str) for category in categories
            ):
                raise DataValidationError(
                    f"catalog[{game_id!r}].categories must be a list of strings"
                )

        return catalog

    @staticmethod
    def _raw_id(value):
        if not isinstance(value, str) or not value.strip():
            raise InvalidParameterError("Use a nonempty original source ID string")
        return value

    def _history(self, user):
        lo = np.searchsorted(self._seen[:, 0], user, side='left')
        hi = np.searchsorted(self._seen[:, 0], user, side='right')
        return self._seen[lo:hi, 1]

    def get_game_details(self, raw_game_id):
        gid = self._raw_id(raw_game_id)
        if gid not in self._item_map:
            raise UnknownItemError(f"Game not present in trained mapping: {gid}")
        if self._catalog_lookup is not None:
            game = self._catalog_lookup(gid)
            if not isinstance(game, GameItem) or game.item_id != gid:
                raise DataValidationError("catalog_lookup must return the matching shared GameItem")
            return game
        row = self._catalog.get(gid, {})
        # Old bundles contain title/image/categories, not verified genres or prices.
        return GameItem(item_id=gid, title=row.get('title') or gid)

    def get_catalog_metadata(self, raw_game_id):
        self.get_game_details(raw_game_id)
        return dict(self._catalog.get(raw_game_id, {}))

    def get_user_played_games(self, raw_user_id):
        user = self._users.get(self._raw_id(raw_user_id))
        if user is None:
            raise UnknownUserError("Source user is absent from the trained mapping")
        return [self.get_game_details(self._items[i]) for i in self._history(user)]

    def recommend_games(self, raw_user_id, top_k=10, exclude_played=True,
                        genre_filter=None, max_price=None, *, extra_seen=()):
        raw_user_id = self._raw_id(raw_user_id)
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
            raise InvalidParameterError("top_k must be a positive integer")
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be boolean")
        if max_price is not None and (isinstance(max_price, bool) or not isinstance(max_price, Real)
                                     or not math.isfinite(max_price) or max_price < 0):
            raise InvalidParameterError("max_price must be finite and nonnegative")
        genres = None
        if genre_filter is not None:
            try:
                values = [genre_filter] if isinstance(genre_filter, str) else list(genre_filter)
            except TypeError as exc:
                raise InvalidParameterError("genre_filter must be a string or collection of strings") from exc
            if not values or any(not isinstance(g, str) or not g.strip() for g in values):
                raise InvalidParameterError("genre_filter must contain nonempty strings")
            genres = {g.strip().casefold() for g in values}
        if isinstance(extra_seen, (str, bytes)):
            raise InvalidParameterError("extra_seen must be a collection of original game IDs")
        try:
            extra_seen = tuple(extra_seen)
        except TypeError as exc:
            raise InvalidParameterError("extra_seen must be a collection of original game IDs") from exc
        if any(not isinstance(g, str) or not g.strip() for g in extra_seen):
            raise InvalidParameterError("extra_seen must contain original game ID strings")
        user = self._users.get(raw_user_id)
        fallback = user is None
        if fallback and not self.allow_cold_start_fallback:
            raise UnknownUserError("No trained profile for this source user; enable popularity fallback explicitly")
        scores = self._popularity.counts if fallback else self._model.score_all(user)
        excluded = set()
        if exclude_played:
            if user is not None:
                excluded.update(self._history(user))
            excluded.update(self._item_map[g] for g in extra_seen if g in self._item_map)
        limit = len(self._items) if genres is not None or max_price is not None else int(top_k)
        result = []
        for i in top_k_indices(scores, limit, excluded):
            game = self.get_game_details(self._items[i])
            if genres is not None and not genres.intersection(g.casefold() for g in game.genres):
                continue
            if max_price is not None and (game.price is None or game.price > max_price):
                continue
            result.append(RecommendationResult(rank=len(result)+1, game_id=game.item_id,
                          title=game.title, genres=game.genres, predicted_score=float(scores[i]),
                          price=game.price, developer=game.developer,
                          model_name='Popularity' if fallback else self.model_name, is_fallback=fallback))
            if len(result) == top_k:
                break
        return result
