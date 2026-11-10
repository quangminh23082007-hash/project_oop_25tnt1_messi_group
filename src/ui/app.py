import streamlit as st

from src.contracts.exceptions import RecSysError
from src.ui.bootstrap import SERVICE_MODES, default_service_mode, get_service
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

    mode_labels = {
        "demo": "Demo",
        "bpr": "Dataset thật - BPR",
        "text-bpr": "Dataset thật - TextBPR",
    }
    default_mode = default_service_mode()
    with st.sidebar:
        st.markdown("### Nguồn dữ liệu")
        service_mode = st.selectbox(
            "Chế độ vận hành",
            options=SERVICE_MODES,
            index=SERVICE_MODES.index(default_mode),
            format_func=mode_labels.__getitem__,
            key="service_mode",
        )
        if service_mode == "demo":
            st.info("Đang dùng dữ liệu minh họa tích hợp sẵn.")
        else:
            st.success(f"Đang dùng model {service_mode.upper()} đã train trên dữ liệu thật.")

    previous_mode = st.session_state.get("_active_service_mode")
    if previous_mode is not None and previous_mode != service_mode:
        for key in tuple(st.session_state):
            if key.startswith(("rec_", "profile_", "catalog_", "benchmark_")):
                del st.session_state[key]
    st.session_state["_active_service_mode"] = service_mode

    try:
        service = get_service(service_mode)
    except (RecSysError, OSError, ValueError) as exc:
        st.error(f"Không thể khởi tạo chế độ {mode_labels[service_mode]}: {exc}")
        st.caption("Hãy kiểm tra model_info.json và weights.npz trong artifacts/models.")
        st.stop()

    st.caption(f"Nguồn hiện tại: {mode_labels[service_mode]} · Model: {service.model_name}")

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
