"""Page 3: Driver vs Road attribution map — scatter, quadrants, operational callouts."""

import streamlit as st
from src.io import ensure_data
from src.layout import inject_full_width
from src.viz import driver_road_scatter

st.markdown("""
<style>
.quad-card {
    background-color: #f8f9fa;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #4e79a7;
    height: 100%;
}
.quad-title {
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 6px;
}
.quad-text {
    font-size: 14px;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Maneuver Prediction & Alert Decision", page_icon="📊", layout="wide", initial_sidebar_state="auto")
    inject_full_width()
    df, score_df, *_ = ensure_data()

    st.markdown("## We can separate driver-induced instability from road-induced roughness.")
    fig = driver_road_scatter(df, score_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Most points cluster near the center, showing that driving is usually stable on normal roads. Uncertainty appears where points spread diagonally upward and right, indicating situations where driver behavior and road conditions overlap, making it hard to attribute instability to a single cause.")

    st.markdown("""
    <style>
    .section-title {
        font-size:22px;
        font-weight:700;
        margin:20px 0 8px 0;
    }

    .qcard {
        background-color:#ffffff;
        padding:16px;
        border-radius:14px;
        box-shadow:0 4px 10px rgba(0,0,0,0.06);
    }

    .qcard.green  { border-left:5px solid #10b981; }
    .qcard.yellow { border-left:5px solid #f59e0b; }
    .qcard.red    { border-left:5px solid #ef4444; }

    .card-title {
        font-weight:600;
        font-size:15px;
        margin-bottom:6px;
    }

    .card-text {
        font-size:14px;
        color:#374151;
        line-height:1.4;
    }

    .pill {
        display:inline-block;
        margin-top:8px;
        padding:3px 10px;
        border-radius:999px;
        font-size:12px;
        background:#f3f4f6;
        color:#111827;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------ Section Header ------------------
    st.markdown('<div class="section-title">Alert Decision Scenarios</div>', unsafe_allow_html=True)
    # ------------------ 2 x 2 Layout ------------------
    row1 = st.columns(2, gap="large")
    row2 = st.columns(2, gap="large")

    # Smooth Driver · Smooth Road
    with row1[0]:
        st.markdown("""
        <div class="qcard green">
            <div class="card-title">Smooth Driver · Smooth Road</div>
            <div class="card-text">
                Controlled driving on a stable surface with predictable vehicle behavior.
            </div>
            <div class="pill">Action: No alert</div>
        </div>
        """, unsafe_allow_html=True)

    # Aggressive Driver · Smooth Road
    with row1[1]:
        st.markdown("""
        <div class="qcard yellow">
            <div class="card-title">Aggressive Driver · Smooth Road</div>
            <div class="card-text">
                Risk is driven by sharp driver inputs despite stable road conditions.
            </div>
            <div class="pill">Action: Driver warning / coaching</div>
        </div>
        """, unsafe_allow_html=True)

    # Smooth Driver · Rough Road
    with row2[0]:
        st.markdown("""
        <div class="qcard yellow">
            <div class="card-title">Smooth Driver · Rough Road</div>
            <div class="card-text">
                Driver remains controlled, but road disturbances affect vehicle stability.
            </div>
            <div class="pill">Action: Road or comfort advisory</div>
        </div>
        """, unsafe_allow_html=True)

    # Aggressive Driver · Rough Road
    with row2[1]:
        st.markdown("""
        <div class="qcard red">
            <div class="card-title">Aggressive Driver · Rough Road</div>
            <div class="card-text">
                Compounding risk from strong driver inputs and poor road response.
            </div>
            <div class="pill">Action: Strong safety alert</div>
        </div>
        """, unsafe_allow_html=True)    

if __name__ == "__main__":
    main()
