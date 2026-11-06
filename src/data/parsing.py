"""Chuẩn hóa ô dữ liệu thô trước khi tạo DTO của contract."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from numbers import Integral
from typing import Iterable, List, Optional

from src.contracts.exceptions import DataValidationError


def is_missing(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and val != val:
        return True
    return type(val).__name__ in {"NAType", "NaTType"}


def clean_id(val: object) -> str:
    """ID về str, bỏ .0 của số thực nguyên và khoảng trắng."""
    if is_missing(val):
        return ""
    if isinstance(val, bool):
        return ""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val).strip()
    text = str(val).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def to_float(value: object, default: float = 0.0) -> float:
    if is_missing(value) or isinstance(value, bool):
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def to_price(value: object) -> Optional[float]:
    """GameItem.price là float hữu hạn >= 0 hoặc None. Thiếu/lỗi/âm thì None (không đoán 0)."""
    if is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B, %Y",
    "%d %b, %Y",
)


def _epoch(moment: datetime) -> int:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        return max(0, int(moment.timestamp()))
    except (OverflowError, OSError, ValueError):
        return 0


def to_timestamp(value: object) -> int:
    """
    timestamp của InteractionRecord: int >= 0 (epoch giây, UTC).

    Nhận số, chuỗi số, datetime/date hoặc chuỗi ngày. Không đọc được hoặc âm thì 0.
    """
    if is_missing(value) or isinstance(value, bool):
        return 0
    if isinstance(value, datetime):
        return _epoch(value)
    if isinstance(value, date):
        return _epoch(datetime(value.year, value.month, value.day))
    if isinstance(value, Integral):
        return max(0, int(value))
    if isinstance(value, float):
        return max(0, int(value)) if math.isfinite(value) else 0

    text = str(value).strip()
    if not text:
        return 0
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return max(0, int(number)) if math.isfinite(number) else 0
    try:
        return _epoch(datetime.fromisoformat(text))
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return _epoch(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return 0


def parse_list(val: object) -> List[str]:
    """genres/tags: JSON, list, ndarray, dict (lấy key) hoặc chuỗi phân tách bằng dấu phẩy."""
    if is_missing(val):
        return []

    if isinstance(val, str):
        text = val.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                items: Iterable[object] = text.split(",")
            else:
                items = parsed if isinstance(parsed, list) else [parsed]
        else:
            items = text.split(",")
    elif isinstance(val, dict):
        items = list(val.keys())
    elif isinstance(val, (list, tuple, set)):
        items = list(val)
    elif hasattr(val, "tolist"):
        converted = val.tolist()
        items = list(converted) if isinstance(converted, (list, tuple)) else [converted]
    else:
        items = [val]

    seen: dict[str, None] = {}
    for item in items:
        if is_missing(item):
            continue
        text = str(item).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            seen.setdefault(text, None)
    return list(seen)


def as_recommended(value: object) -> bool | None:
    """True/False cho Recommended và Not Recommended. Giá trị lạ trả về None."""
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"recommended", "true", "1", "yes"}:
        return True
    if text in {"not recommended", "false", "0", "no"}:
        return False
    return None


def label_from_recommended(recommended: bool) -> float:
    """Nhãn nhị phân của contract: Recommended = 1.0, Not Recommended = 0.0."""
    return 1.0 if recommended else 0.0


def label_from_rating(value: object) -> float:
    """Cột rating có sẵn phải đã là nhị phân 0/1; thang khác thì báo lỗi, không tự đoán ngưỡng."""
    if isinstance(value, bool):
        return label_from_recommended(value)
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise DataValidationError(f"rating phải là số 0 hoặc 1, nhận {value!r}.") from None
    if number in (0.0, 1.0):
        return number
    raise DataValidationError(
        f"rating phải là nhãn nhị phân 0/1 theo contract, nhận {value!r}. "
        "Hãy xuất lại file với cột 'recommended' hoặc rating đã nhị phân."
    )


def wilson_score(positive: int, negative: int, z: float = 1.96) -> float:
    """Wilson score lower bound, thang 0.0–1.0. Chưa có đánh giá thì 0.5."""
    total = positive + negative
    if total <= 0:
        return 0.5
    p = positive / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, min(1.0, (center - margin) / denominator))
