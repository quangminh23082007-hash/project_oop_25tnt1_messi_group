from pathlib import Path

import pytest

import src.integration as integration
from src.contracts.exceptions import ConfigurationError


class FakePretrainedService:
    model_name = "bpr"
    allow_cold_start_fallback = True


def _write_bundle_files(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "model_info.json").write_text("{}", encoding="utf-8")
    (folder / "weights.npz").write_bytes(b"weights")


def test_real_factory_wraps_pretrained_service(monkeypatch, tmp_path):
    model_dir = tmp_path / "bpr" / "cpu"
    _write_bundle_files(model_dir)
    loaded = {}

    monkeypatch.setattr(integration, "model_dir_for", lambda model_kind: model_dir)

    def fake_load(folder, *, allow_cold_start_fallback):
        loaded["folder"] = Path(folder)
        loaded["fallback"] = allow_cold_start_fallback
        return FakePretrainedService()

    monkeypatch.setattr(integration.PretrainedServiceFacade, "load", fake_load)

    service = integration.create_real_service("bpr")

    assert loaded == {"folder": model_dir, "fallback": True}
    assert service.mode == "bpr"
    assert service.model_name == "bpr"
    assert service.is_ready is True
    assert service.supports_catalog_filters is False


def test_real_factory_reports_missing_bundle(monkeypatch, tmp_path):
    model_dir = tmp_path / "missing" / "cpu"
    monkeypatch.setattr(integration, "model_dir_for", lambda model_kind: model_dir)

    with pytest.raises(ConfigurationError, match="Chưa cài model BPR"):
        integration.create_real_service("bpr")
