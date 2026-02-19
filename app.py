"""
Entry point: run as streamlit run app.py
Uses st.navigation and st.Page so the sidebar shows only the defined pages (no separate "app" entry).
"""

import streamlit as st
from src.io import get_data
from src.scores import quality_checks
from src.layout import inject_full_width


def main():
    st.set_page_config(
        page_title="Maneuver Prediction & Alert Decision",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_full_width()

    df, score_df, class_summary_df = get_data()
    qc = quality_checks(df, score_df)
    st.session_state["df"] = df
    st.session_state["score_df"] = score_df
    st.session_state["class_summary_df"] = class_summary_df
    st.session_state["qc"] = qc

    pages = [
        st.Page("pages/introduction.py", title="Welcome to the Drive"),
        st.Page("pages/context_and_objectives.py", title="The Problem We’re Solving"),        
        st.Page("pages/5_about_data.py", title="Domain Understanding"),
        st.Page("pages/data_understanding.py", title="Data Understanding"),
        st.Page("pages/4_same_maneuver_not_same_behavior.py", title="Same Move, Different Risk"),
        st.Page("pages/3_driver_vs_road_attribution.py", title="Driver or Road Who’s Responsible?"),
        st.Page("pages/8_Decision_Engine.py", title="How Alerts Are Decided?"),
        st.Page("pages/6_Maneuver_Prediction.py", title="Can We Predict the Move?"),
        st.Page("pages/9_Raw_to_Alerts_Pipeline.py", title="What Happens After the Sensors Trigger?"),
    ]
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
