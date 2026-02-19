import streamlit as st

CARD_CSS = """
<style>
.obj-card {
    background-color: #ffffff;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #1a6b6b;
    margin-bottom: 1rem;
}
.obj-card-title {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    color: #1a6b6b;
}
.obj-card-text {
    font-size: 0.95rem;
    color: #333;
    line-height: 1.45;
}
.obj-section-title {
    font-size: 1.17rem;
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
    color: inherit;
}
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)
st.markdown("## Project Overview")

problem_text = "Raw IMU sensor data captures vehicle motion but cannot directly indicate driving maneuvers or distinguish between driver-induced and road-induced risk, making it difficult to generate accurate alerts."
project_text = "This project turns raw IMU sensor data into meaningful driving insights by aggregating signals into statistical features, predicting driving maneuvers, and using a decision engine to distinguish driver-induced risk from road-induced disturbance, ultimately generating clear, context-aware alerts."

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div class="obj-card">
            <div class="obj-card-title">Problem Statement</div>
            <div class="obj-card-text">{problem_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="obj-card">
            <div class="obj-card-title">What this project does</div>
            <div class="obj-card-text">{project_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("#### What each page covers")

page_cards = [

    (
        "Domain Understanding",
        "Explains the problem domain: vehicle motion axes (AccX/Y/Z, GyroX/Y/Z), how the IMU signals relate to real behavior.",
    ),
    (
        "Data Understanding",
        "Shows how IMU data is transformed to enable maneuver prediction and alert decisions. Covers data quality checks, feature engineering, and creation of axis-level instability signals.",
    ),
    (
        "Same Maneuver ≠ Same Behavior",
        "Shows that the same maneuver can have very different risk intensity. Connects driver instability, road instability, and yaw behavior to demonstrate that risk depends on context, not just labels.",
    ),
    (
        "Driver vs Road Attribution",
        "Separates driver-induced instability (AccX, AccY, GyroZ) from road-induced instability (AccZ), enabling clear attribution of risk to driving behavior versus road conditions.",
    ),
    (
        "Decision Engine",
        "Uses normalized Driver Instability and Road Instability (0–1) to apply ordered rules. Level 0 returns no alert, Levels 1–3 produce cause, severity, and a message.",
    ),
    (
        "Maneuver Prediction",
        "Trains a Random Forest to classify driving maneuvers. Shows test accuracy, per-class metrics, confusion matrix, and feature importance.",
    ),
    (
        "Raw-to-Alerts Pipeline",
        "Runs the full flow from raw_sensor_data.csv to alerts. Lets you select a raw record, view it, and see the pipeline output (predictions and alert) for that TaskID.",
    ),
]

for title, text in page_cards:
    st.markdown(
        f"""
        <div class="obj-card">
            <div class="obj-card-title">{title}</div>
            <div class="obj-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
