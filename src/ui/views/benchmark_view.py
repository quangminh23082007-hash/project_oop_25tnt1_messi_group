from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import streamlit as st

from src.ui.components.error_handler import handle_ui_error
from src.ui.components.game_card import render_recommendation_card
from src.ui.components.metric_table import render_metric_table
from src.ui.data import user_id_options
from src.ui.service_protocol import UIRecommendationService


def _summary_path() -> Path:
    configured = os.getenv("RECSYS_BENCHMARK_SUMMARY", "").strip()
    if configured:
        return Path(configured)
    project_root = Path(__file__).resolve().parents[3]
    benchmark_summary = project_root / "results" / "benchmark" / "summary.csv"
    if benchmark_summary.is_file():
        return benchmark_summary
    candidates = sorted(
        (
            path
            for path in (project_root / "results" / "benchmark").glob("*/summary.csv")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return project_root / "results" / "summary.csv"


def _load_summary() -> list[dict[str, Any]]:
    path = _summary_path()
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                clean_key = key.strip()
                try:
                    row[clean_key] = float(value) if value not in (None, "") else value
                except (TypeError, ValueError):
                    row[clean_key] = value
            rows.append(row)
        return rows


def _metric_key(rows: list[dict[str, Any]], target: str) -> str | None:
    if not rows:
        return None
    normalized = target.casefold().replace(" ", "").replace("_", "")
    return next(
        (
            key for key in rows[0]
            if key.casefold().replace(" ", "").replace("_", "") == normalized
        ),
        None,
    )


@handle_ui_error
def render_benchmark_view(service: UIRecommendationService) -> None:
    st.subheader("Quản lý mô hình và đối chuẩn")

    st.markdown('<div class="section-label">Thao tác nhanh trên Service</div>', unsafe_allow_html=True)
    col_info, col_toggle = st.columns([2, 1])
    with col_info:
        st.markdown(f"**Mô hình hiện tại:** `{service.model_name}`")
        st.markdown(f"**Trạng thái:** `{'SẴN SÀNG' if service.is_ready else 'CHƯA SẴN SÀNG'}`")
    with col_toggle:
        current = service.allow_cold_start_fallback
        enabled = st.toggle("Bật fallback cold-start", value=current)
        if enabled != current:
            service.allow_cold_start_fallback = enabled
            st.rerun()

    st.divider()
    st.markdown('<div class="section-label">Quantitative Benchmark</div>', unsafe_allow_html=True)
    rows = _load_summary()
    if not rows:
        st.info("Chưa tìm thấy results/summary.csv. Hãy chạy benchmark offline để nạp dữ liệu.")
    else:
        render_metric_table(rows)
        model_key = next(
            (key for key in rows[0] if key.casefold() in {"model", "model_name", "name"}),
            None,
        )
        ndcg_key = _metric_key(rows, "NDCG@5")
        if model_key and ndcg_key:
            chart_rows = [
                {"Mô hình": str(row[model_key]), "NDCG@5": row[ndcg_key]}
                for row in rows
                if isinstance(row.get(ndcg_key), (int, float))
            ]
            if chart_rows:
                st.markdown("**So sánh NDCG@5**")
                st.bar_chart(chart_rows, x="Mô hình", y="NDCG@5")

    st.divider()
    st.markdown('<div class="section-label">So sánh định tính trên một user</div>', unsafe_allow_html=True)
    available_users = user_id_options(service)
    if available_users:
        options = [*available_users, "Nhập User ID khác..."]
        selected_user = st.selectbox("User ID gốc", options=options, key="benchmark_user_select")
        if selected_user == "Nhập User ID khác...":
            raw_user_id = st.text_input("User ID tùy chỉnh", key="benchmark_user_custom", placeholder="Nhập raw_user_id")
        else:
            raw_user_id = selected_user
    else:
        raw_user_id = st.text_input("User ID gốc", key="benchmark_user_input", placeholder="Nhập raw_user_id")
    compare = st.button("So sánh đề xuất", type="primary")
    if not compare:
        return

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Đề xuất cá nhân hóa**")
        personalized = service.recommend_games(raw_user_id=raw_user_id, top_k=5)
        if not personalized:
            st.info("Không có đề xuất cá nhân hóa.")
        for result in personalized:
            render_recommendation_card(result)

    with right:
        st.markdown("**Đề xuất phổ biến (MostPop baseline)**")
        baseline_provider = getattr(service, "recommend_baseline", None)
        if not callable(baseline_provider):
            st.info(
                "Baseline phổ biến chưa được cung cấp bởi Service contract. "
                "UI không tự tạo user giả hoặc truy cập ID nội bộ."
            )
            return
        baseline = baseline_provider(raw_user_id=raw_user_id, top_k=5)
        if not baseline:
            st.info("Không có đề xuất phổ biến.")
        for result in baseline:
            render_recommendation_card(result)
