import streamlit as st

from src.service.recommendation_service import RecommendationService
from src.ui.components.error_handler import handle_ui_error
from src.ui.components.game_card import render_recommendation_card
from src.ui.data import genre_options, user_id_options


@handle_ui_error
def render_recommend_view(service: RecommendationService) -> None:
    st.subheader("Đề xuất game")

    col_ctrl, col_results = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="section-label">Bộ điều khiển</div>', unsafe_allow_html=True)
        available_users = user_id_options(service)
        if available_users:
            options = [*available_users, "Nhập User ID khác..."]
            selected_user = st.selectbox("User ID gốc", options=options, key="rec_user_select")
            if selected_user == "Nhập User ID khác...":
                raw_user_id = st.text_input("User ID tùy chỉnh", value="", placeholder="Nhập raw_user_id", key="rec_user_custom")
            else:
                raw_user_id = selected_user
        else:
            raw_user_id = st.text_input("User ID gốc", value="", placeholder="Nhập raw_user_id", key="rec_user_input")
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        exclude_played = st.checkbox("Loại bỏ game đã chơi", value=True)

        available_genres = genre_options(service)
        selected_genre = st.selectbox("Thể loại", options=["Tất cả", *available_genres])
        genre_filter = None if selected_genre == "Tất cả" else selected_genre

        max_price_input = st.number_input("Giá tối đa (USD)", min_value=0.0, value=0.0, step=5.0)
        max_price = max_price_input if max_price_input > 0.0 else None

        submit = st.button("Lấy đề xuất", use_container_width=True, type="primary")

    with col_results:
        if submit:
            results = service.recommend_games(
                raw_user_id=raw_user_id,
                top_k=top_k,
                exclude_played=exclude_played,
                genre_filter=genre_filter,
                max_price=max_price,
            )

            if not results:
                st.info("Không tìm thấy game nào phù hợp với tiêu chí lọc.")
                return

            if results[0].is_fallback:
                st.markdown(
                    f'<div class="fallback-box">'
                    f'<strong>[THONG BAO FALLBACK]</strong> Người dùng chưa có dữ liệu tích cực hoặc là người dùng mới. '
                    f'Hệ thống đang hiển thị đề xuất từ mô hình phổ biến (<code>{results[0].model_name}</code>).'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"**{len(results)} đề xuất phù hợp**")
            for r in results:
                render_recommendation_card(r)
        else:
            st.info("Nhập User ID và bấm 'Lấy đề xuất' để nhận danh sách game phù hợp.")
