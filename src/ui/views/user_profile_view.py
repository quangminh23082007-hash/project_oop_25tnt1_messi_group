import streamlit as st

from src.service.recommendation_service import RecommendationService
from src.ui.components.error_handler import handle_ui_error
from src.ui.data import user_id_options


@handle_ui_error
def render_user_profile_view(service: RecommendationService) -> None:
    st.subheader("Hồ sơ người chơi")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        available_users = user_id_options(service)
        if available_users:
            options = [*available_users, "Nhập User ID khác..."]
            selected_user = st.selectbox("User ID gốc", options=options, key="profile_user_select")
            if selected_user == "Nhập User ID khác...":
                raw_user_id = st.text_input("User ID tùy chỉnh", key="profile_user_custom", placeholder="Nhập raw_user_id")
            else:
                raw_user_id = selected_user
        else:
            raw_user_id = st.text_input("User ID gốc", key="profile_user_input", placeholder="Nhập raw_user_id")
    with col_btn:
        st.write("")
        st.write("")
        submit = st.button("Kiểm tra lịch sử", use_container_width=True, type="primary")

    if submit:
        games = service.get_user_played_games(raw_user_id)
        if not games:
            st.info(f"Người dùng '{raw_user_id}' chưa có lịch sử tương tác nào trong hệ thống.")
            return

        st.markdown(f"**Tổng số game đã tương tác:** `{len(games)} game`")
        st.caption("Bao gồm cả đánh giá tích cực và tiêu cực.")
        st.dataframe(
            [
                {
                    "Mã game": g.item_id,
                    "Tên game": g.title,
                    "Thể loại": ", ".join(g.genres) or "N/A",
                    "Nhà phát triển": g.developer,
                    "Giá": "Miễn phí" if g.price == 0.0 else (f"${g.price:.2f}" if g.price is not None else "N/A"),
                }
                for g in games
            ],
            use_container_width=True,
            hide_index=True,
        )
