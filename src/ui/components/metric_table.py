from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st


def render_metric_table(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        st.info("Chưa có dữ liệu benchmark trong results/summary.csv.")
        return
    st.dataframe(list(rows), use_container_width=True, hide_index=True)
