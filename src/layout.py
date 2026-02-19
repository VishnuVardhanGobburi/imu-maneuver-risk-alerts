"""Shared layout: full-width container CSS for Streamlit."""

import streamlit as st


def inject_full_width():
    """Inject CSS so main content uses full width on load and after refresh (Streamlit 1.38+ and 1.40+)."""
    st.markdown(
        """
        <style>
        /* Full width: target every known main-content wrapper */
        [data-testid="stAppViewContainer"] main .block-container,
        [data-testid="stAppViewContainer"] section[data-testid="stMain"] > div,
        section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"],
        div[data-testid="stMainBlockContainer"],
        .appview-container .main .block-container,
        .main .block-container {
            max-width: 100%% !important;
            width: 100%% !important;
            padding-top: 0.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 1rem !important;
        }
        .appview-container .main,
        section[data-testid="stMain"] {
            max-width: 100%% !important;
            padding-top: 0 !important;
        }
        /* Ensure app view container doesn't constrain */
        [data-testid="stAppViewContainer"] {
            max-width: 100%% !important;
        }
        /* Pull main content up: less top margin */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            padding-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
