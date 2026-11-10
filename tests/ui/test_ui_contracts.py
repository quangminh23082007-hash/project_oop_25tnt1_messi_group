import pytest
from src.contracts.dto import GameItem, RecommendationResult
from src.contracts.exceptions import (
    ColdStartError,
    UnknownItemError,
    UnknownUserError,
)
from src.ui.bootstrap import (
    SERVICE_MODES,
    create_demo_service,
    default_service_mode,
    get_service,
)
from src.ui.components.error_handler import handle_ui_error
from src.ui.components.game_card import get_steam_poster_url
from src.ui.data import game_id_options, genre_options, user_id_options


def test_bootstrap_service_contract():
    service = create_demo_service()
    assert service.is_ready
    assert service.model_name != ""
    assert service.allow_cold_start_fallback is True


def test_mode_aware_bootstrap_defaults_to_demo(monkeypatch):
    monkeypatch.delenv("RECSYS_MODE", raising=False)
    assert SERVICE_MODES == ("demo", "bpr", "text-bpr")
    assert default_service_mode() == "demo"

    service = get_service("demo")
    assert service.mode == "demo"
    assert service.is_ready
    assert service.supports_catalog_filters is True


def test_invalid_environment_mode_falls_back_to_demo(monkeypatch):
    monkeypatch.setenv("RECSYS_MODE", "not-a-model")
    assert default_service_mode() == "demo"


def test_ui_dynamic_data_discovery():
    service = create_demo_service()
    users = user_id_options(service)
    genres = genre_options(service)
    games = game_id_options(service)

    assert len(users) > 0
    assert "steam_user_101" in users
    assert len(genres) > 0
    assert "Action" in genres
    assert len(games) > 0
    assert "570" in games


def test_recommend_baseline_contract():
    service = create_demo_service()
    baseline = service.recommend_baseline(raw_user_id="steam_user_101", top_k=3)
    assert len(baseline) <= 3
    for res in baseline:
        assert isinstance(res, RecommendationResult)
        assert res.is_fallback is True
        assert res.rank >= 1
        assert res.game_id != ""


def test_steam_poster_url_generation():
    url = get_steam_poster_url("570")
    assert url == "https://cdn.akamai.steamstatic.com/steam/apps/570/header.jpg"

    url_trimmed = get_steam_poster_url("  730 \n")
    assert url_trimmed == "https://cdn.akamai.steamstatic.com/steam/apps/730/header.jpg"


def test_handle_ui_error_catches_recsys_exceptions():
    @handle_ui_error
    def raise_unknown_user():
        raise UnknownUserError("User not found")

    @handle_ui_error
    def raise_cold_start():
        raise ColdStartError("No positive interactions")

    @handle_ui_error
    def raise_unknown_item():
        raise UnknownItemError("Game not found")

    raise_unknown_user()
    raise_cold_start()
    raise_unknown_item()
