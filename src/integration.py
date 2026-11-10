from __future__ import annotations

from pathlib import Path

from src.contracts.exceptions import ConfigurationError
from src.service.pretrained import PretrainedServiceFacade
from src.ui.service_adapter import UIServiceAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDERS = {
    "bpr": "bpr",
    "text-bpr": "text_bpr",
}


def model_dir_for(model_kind: str) -> Path:
    try:
        folder_name = MODEL_FOLDERS[model_kind]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_FOLDERS))
        raise ConfigurationError(
            f"Chế độ model '{model_kind}' không được hỗ trợ. Hãy chọn: {supported}."
        ) from exc
    return PROJECT_ROOT / "artifacts" / "models" / folder_name / "cpu"


def create_real_service(model_kind: str = "bpr") -> UIServiceAdapter:
    model_dir = model_dir_for(model_kind)
    required = (model_dir / "model_info.json", model_dir / "weights.npz")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        archive_name = (
            "text-bpr-cpu-legacy-v1.zip"
            if model_kind == "text-bpr"
            else "bpr-cpu-legacy-v1.zip"
        )
        raise ConfigurationError(
            f"Chưa cài model {model_kind.upper()} tại '{model_dir}'. "
            f"Thiếu: {', '.join(missing)}. Hãy cài từ {archive_name} trước."
        )

    facade = PretrainedServiceFacade.load(
        model_dir,
        allow_cold_start_fallback=True,
    )
    # Legacy bundles contain titles/images but do not provide verified genre,
    # price and developer fields used by the UI filters.
    return UIServiceAdapter(
        facade,
        mode=model_kind,
        supports_catalog_filters=False,
    )


__all__ = ["create_real_service", "model_dir_for"]
