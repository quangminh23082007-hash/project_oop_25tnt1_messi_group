import streamlit as st

from src.ui.components.error_handler import handle_ui_error
from src.ui.components.game_card import render_game_card
from src.ui.data import game_id_options
from src.ui.service_protocol import UIRecommendationService


@handle_ui_error
def render_game_catalog_view(service: UIRecommendationService) -> None:
    st.subheader("Tra cứu game catalog")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        available_games = game_id_options(service)
        if available_games:
            options = [*available_games, "Nhập mã game khác..."]
            selected_game = st.selectbox(
                "Mã game gốc (Steam App ID)",
                options=options,
                key="catalog_game_select",
            )
            if selected_game == "Nhập mã game khác...":
                raw_game_id = st.text_input(
                    "Mã game tùy chỉnh",
                    value="",
                    placeholder="Nhập Steam App ID (ví dụ: 570)",
                    key="catalog_game_custom",
                )
            else:
                raw_game_id = selected_game
        else:
            raw_game_id = st.text_input(
                "Mã game gốc (Steam App ID)",
                value="",
                placeholder="Nhập Steam App ID",
                key="catalog_game_input",
            )
    with col_btn:
        st.write("")
        st.write("")
        submit = st.button("Xem chi tiết", use_container_width=True, type="primary")

    if submit:
        game = service.get_game_details(raw_game_id)
        st.markdown("**Thông tin game**")
        render_game_card(game)
