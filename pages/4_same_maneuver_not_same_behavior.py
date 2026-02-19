"""Page 4: Same maneuver ≠ same behavior — 4 tabs: Driver/Road instability, Bias, Connecting the dots."""

import streamlit as st
import pandas as pd
from src.io import ensure_data
from src.layout import inject_full_width
from src.viz import (
    violin_driver_instability,
    violin_road_instability,
    bias_violin_turning_only,
    TARGET_LABELS,
)


def main():
    st.set_page_config(page_title="Maneuver Prediction & Alert Decision", page_icon="📊", layout="wide", initial_sidebar_state="auto")
    inject_full_width()
    df, score_df, *_ = ensure_data()

    st.markdown("## Within the same maneuver, risk intensity varies a lot.")
    st.markdown(
        "<style>div[data-testid='stTabs'] button { min-width: 7rem; }</style>",
        unsafe_allow_html=True,
    )

    tab_driver, tab_road, tab_bias, tab_dots = st.tabs([
        "Driver instability by class",
        "Road Instability by class",
        "Gyro Directional Bias (turns only)",
        "Connecting the dots",
    ])

    with tab_driver:
        fig1 = violin_driver_instability(df, score_df)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("Overall, the plot shows that acceleration is generally smooth, left and right turns are more variable with left turns being the most unstable, and braking consistently produces high instability.")

    with tab_road:
        fig2 = violin_road_instability(df, score_df)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("The plot shows that vertical vehicle response reflects interaction between road conditions and driving maneuvers, with acceleration and braking amplifying vertical effects more than turns.")

    with tab_bias:
        bias_col = "GyroZ_Directional_Bias" if "GyroZ_Directional_Bias" in score_df.columns else "AccY_Directional_Bias"
        fig3 = bias_violin_turning_only(df, score_df, score_col=bias_col)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("While left and right turns are directionally opposite and generally consistent, left turns exhibit greater extremes in yaw response, suggesting they are often executed more aggressively or with higher steering variability than right turns.")

    with tab_dots:
        # Reduce gap between heading and image
        st.markdown(
            "<style>"
            "div[data-testid='stVerticalBlock'] > div { padding-top: 0.15rem !important; padding-bottom: 0.15rem !important; }"
            "div[data-testid='stImage'] { margin-top: -0.9rem !important; }"
            "</style>",
            unsafe_allow_html=True,
        )
        st.markdown('<h4 style="margin: 0 0 0 0;">Bringing the three analysis together</h4>', unsafe_allow_html=True)
        st.image("Images/connecting_the_dots.png", width=1000)
        st.markdown("Together, these results show that the same maneuver can shift from low risk to high risk depending on driver input strength, road interaction, and steering intensity, reinforcing that risk must be assessed by intensity and context, not by maneuver type alone.")


main()
