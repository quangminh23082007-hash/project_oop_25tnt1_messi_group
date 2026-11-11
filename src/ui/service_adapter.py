from __future__ import annotations

from typing import Any


class UIServiceAdapter:
    """Normalize demo and pretrained services for the Streamlit UI."""

    def __init__(
        self,
        service: Any,
        *,
        mode: str,
        supports_catalog_filters: bool,
    ) -> None:
        self._service = service
        self.mode = mode
        self._supports_catalog_filters = supports_catalog_filters

    @property
    def model_name(self) -> str:
        return str(self._service.model_name)

    @property
    def is_ready(self) -> bool:
        return bool(getattr(self._service, "is_ready", True))

    @property
    def allow_cold_start_fallback(self) -> bool:
        return bool(self._service.allow_cold_start_fallback)

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        self._service.allow_cold_start_fallback = value

    @property
    def supports_catalog_filters(self) -> bool:
        return self._supports_catalog_filters

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)


__all__ = ["UIServiceAdapter"]
