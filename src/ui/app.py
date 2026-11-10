import streamlit as st

from src.ui.bootstrap import get_service
from src.ui.styles import load_clean_css
from src.ui.views.benchmark_view import render_benchmark_view
from src.ui.views.game_catalog_view import render_game_catalog_view
from src.ui.views.recommend_view import render_recommend_view
from src.ui.views.user_profile_view import render_user_profile_view


def main() -> None:
    st.set_page_config(
        page_title="Steam Game Recommender System",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_clean_css()

    st.title("HỆ THỐNG ĐỀ XUẤT GAME")
    st.caption("Lớp 25TNT1 - Nhóm Messi")

    service = get_service()

    tab_rec, tab_prof, tab_cat, tab_bench = st.tabs([
        "Đề xuất Game",
        "Hồ sơ Người chơi",
        "Danh mục Catalog",
        "Quản lý & Đối chuẩn",
    ])

    with tab_rec:
        render_recommend_view(service)

    with tab_prof:
        render_user_profile_view(service)

    with tab_cat:
        render_game_catalog_view(service)

    with tab_bench:
        render_benchmark_view(service)


if __name__ == "__main__":
    main()
