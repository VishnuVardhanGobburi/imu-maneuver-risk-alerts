"""Page 5: Domain Understanding — KPIs, data quality checks, domain background."""

import streamlit as st
import pandas as pd
from src.io import ensure_data
from src.layout import inject_full_width


# ---------- CSS for cards ----------
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    height: 100%;
    border: 1px solid #d1d5db;
}
.card-title {
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 8px;
}
.card-text {
    font-size: 14px;
    color: #333333;
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    margin: 20px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Maneuver Prediction & Alert Decision", page_icon="📊", layout="wide", initial_sidebar_state="auto")
    inject_full_width()
    df, *_ = ensure_data()

    st.markdown("## Domain Understanding")


    st.markdown("Two sensors, an accelerometer and a gyrometer, capture vehicle motion. The accelerometer measures linear movement, while the gyrometer measures rotation. Together, they sense motion across three axes X, Y, and Z representing forward and backward, side to side, and up and down movement, providing a 360° view of vehicle dynamics.")
    # ---------- Accelerometer ----------
    st.markdown('<div class="section-title">Accelerometer (Linear Motion)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">X-axis (Longitudinal) → Throttle Force</div>
            <div class="card-text">
                Forward/backward push from acceleration and braking.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Y-axis (Lateral) → Cornering Force</div>
            <div class="card-text">
                Sideways force during turns and lane changes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Z-axis (Vertical) → Road Impact</div>
            <div class="card-text">
                Bumps, potholes, uneven surfaces, lift and drop.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Gyrometer ----------
    st.markdown('<div class="section-title">Gyrometer (Rotational Motion)</div>', unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="card">
            <div class="card-title">X-axis (Roll) → Vehicle Lean</div>
            <div class="card-text">
                Body roll during turns.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="card">
            <div class="card-title">Y-axis (Pitch) → Nose Motion</div>
            <div class="card-text">
                Nose dive on braking, lift on acceleration
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="card">
            <div class="card-title">Z-axis (Yaw) → Steering Direction</div>
            <div class="card-text">
                Left/right turning control.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Domain Knowledge Image ----------
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### How the Car Moves")
    st.image(
        "Images/domain_knowledge_image.png",
        width=700
    )
    st.markdown("Below are the references used to understand the domain knowledge")
    st.markdown(""" 
- https://lastminuteengineers.com/mpu6050-accel-gyro-arduino-tutorial/?utm_source=chatgpt.com
- https://www.youtube.com/watch?v=1GEQwWGUEBI
""")




if __name__ == "__main__":
    main()
