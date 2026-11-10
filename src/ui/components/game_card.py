import streamlit as st

from src.contracts.dto import GameItem, RecommendationResult


def get_steam_poster_url(game_id: str) -> str:
    cleaned = str(game_id).strip()
    return f"https://cdn.akamai.steamstatic.com/steam/apps/{cleaned}/header.jpg"


def _render_poster(game_id: str, title: str) -> None:
    poster_url = get_steam_poster_url(game_id)
    st.markdown(
        f'<div class="poster-frame"><img src="{poster_url}" alt="{title}" '
        'onerror="this.style.display=\'none\'; this.parentElement.classList.add(\'poster-missing\');">'
        '<span>[NO POSTER]</span></div>',
        unsafe_allow_html=True,
    )


def render_recommendation_card(result: RecommendationResult) -> None:
    genres_str = ", ".join(result.genres) if result.genres else "N/A"
    if result.price is None:
        price_str = "Gia: N/A"
    elif result.price == 0.0:
        price_str = "Gia: Mien phi (Free)"
    else:
        price_str = f"Gia: ${result.price:.2f}"

    col_img, col_info = st.columns([1, 2.5])
    with col_img:
        _render_poster(result.game_id, result.title)

    with col_info:
        st.markdown(
            f'<span class="badge-rank">#{result.rank}</span> '
            f'<strong style="font-size: 1.15rem; color: #e6edf3; margin-left: 6px;">{result.title}</strong>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="meta-text">'
            f'ID: <code>{result.game_id}</code> | Studio: {result.developer}<br>'
            f'The loai: {genres_str}<br>'
            f'<strong>{price_str}</strong> | Diem phu hop: <code>{result.predicted_score:.4f}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr style="border-color: #2c3542; margin: 8px 0 16px 0;">', unsafe_allow_html=True)


def render_game_card(game: GameItem) -> None:
    genres_str = ", ".join(game.genres) if game.genres else "N/A"
    if game.price is None:
        price_str = "Gia: N/A"
    elif game.price == 0.0:
        price_str = "Gia: Mien phi (Free)"
    else:
        price_str = f"Gia: ${game.price:.2f}"

    col_img, col_info = st.columns([1, 2.5])
    with col_img:
        _render_poster(game.item_id, game.title)

    with col_info:
        st.markdown(
            f'<strong style="font-size: 1.15rem; color: #e6edf3;">{game.title}</strong>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="meta-text">'
            f'ID: <code>{game.item_id}</code> | Studio: {game.developer}<br>'
            f'The loai: {genres_str}<br>'
            f'<strong>{price_str}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr style="border-color: #2c3542; margin: 8px 0 16px 0;">', unsafe_allow_html=True)
