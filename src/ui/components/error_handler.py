from functools import wraps
from typing import Any, Callable
import streamlit as st

from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    RecSysError,
    UnknownItemError,
    UnknownUserError,
)


def handle_ui_error(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except UnknownUserError as exc:
            st.error(f"[LOI NGUOI DUNG] {exc}")
            st.caption("Goi y: Kiem tra lai User ID hoac bat che do Fallback tai muc Dieu khien.")
        except ColdStartError as exc:
            st.warning(f"[COLD-START] {exc}")
            st.caption("Goi y: Nguoi dung chua co tuong tac tich cuc. Hay bat che do Fallback de nhan goi y dai tra.")
        except UnknownItemError as exc:
            st.error(f"[LOI GAME] {exc}")
        except InvalidParameterError as exc:
            st.warning(f"[THAM SO KHONG HOP LE] {exc}")
        except ModelNotFittedError as exc:
            st.error(f"[MO HINH CHUA SAN SANG] {exc}. Vui long huan luyen mo hinh truoc khi goi y.")
        except DataValidationError as exc:
            st.error(f"[DU LIEU KHONG HOP LE] {exc}")
        except ConfigurationError as exc:
            st.error(f"[LOI CAU HINH] {exc}")
        except EmptyDatasetError as exc:
            st.error(f"[DU LIEU RONG] {exc}")
        except RecSysError as exc:
            st.error(f"[LOI HE THONG] {exc}")
        except Exception as exc:
            st.error(f"[LOI BAT NGO] {exc}")
    return wrapper
