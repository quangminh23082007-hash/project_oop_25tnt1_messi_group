import streamlit as st


def load_clean_css() -> None:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0f1318;
            color: #e6edf3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        [data-testid="stHeader"] {
            background-color: #0f1318;
        }

        [data-testid="stSidebar"] {
            background-color: #181e26;
            border-right: 1px solid #2c3542;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .item-card {
            background-color: #181e26;
            border: 1px solid #2c3542;
            border-radius: 4px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        .poster-img {
            width: 100%;
            max-width: 220px;
            border-radius: 3px;
            border: 1px solid #2c3542;
        }

        .poster-placeholder {
            width: 100%;
            max-width: 220px;
            height: 102px;
            background-color: #222933;
            border: 1px dashed #3d4756;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #8b949e;
            font-size: 0.8rem;
        }

        .poster-frame {
            width: 100%;
            max-width: 220px;
            min-height: 102px;
            background-color: #222933;
            border: 1px solid #2c3542;
            border-radius: 3px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .poster-frame img {
            display: block;
            width: 100%;
            height: auto;
        }

        .poster-frame span {
            display: none;
            color: #8b949e;
            font-size: 0.8rem;
        }

        .poster-frame.poster-missing span {
            display: block;
        }

        .poster-placeholder span {
            color: #8b949e;
        }

        .meta-text {
            color: #8b949e;
            font-size: 0.85rem;
            margin-top: 4px;
        }

        .fallback-box {
            background-color: #231c0e;
            border: 1px solid #d29922;
            border-radius: 4px;
            padding: 10px 14px;
            color: #d29922;
            font-size: 0.9rem;
            margin-bottom: 14px;
        }

        .badge-rank {
            background-color: #1f6feb;
            color: #ffffff;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85rem;
            display: inline-block;
        }

        .section-label {
            color: #8b949e;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        div.stButton > button[kind="primary"] {
            background-color: #1f6feb;
            border: 1px solid #1f6feb;
            color: #ffffff;
            border-radius: 4px;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #1f6feb;
            border-color: #e6edf3;
        }
    </style>
    """, unsafe_allow_html=True)
