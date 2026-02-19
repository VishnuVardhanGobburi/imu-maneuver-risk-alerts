import streamlit as st
import pandas as pd
from src.io import ensure_data, load_raw_sensor
from src.layout import inject_full_width
from src.viz import maneuver_distribution, scorecard_grouped_bars, axis_distribution_histograms, TARGET_LABELS

AXES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
SCORE_COLS = ["Intensity", "Variability", "Impulsiveness", "Directional_Bias"]

# Same card style as Domain Understanding: white background, shadow, rounded; colored border in Preprocessing tab
CARD_CSS = """
<style>
.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    height: 100%;
    border: 2px solid #1a6b6b;
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
"""

def main():
    st.set_page_config(page_title="Data Understanding", page_icon="📋", layout="wide", initial_sidebar_state="auto")
    inject_full_width()
    st.markdown(
        "<style>div[data-testid='stTabs'] button { min-width: 8rem; padding-left: 0.75rem; padding-right: 0.75rem; }</style>",
        unsafe_allow_html=True,
    )
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    try:
        df, score_df, *_ = ensure_data()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return
    if df is None or df.empty:
        st.warning("No data loaded. Ensure the app has run and features data is available.")
        return

    # Tabs: Quality & Features, Preprocessing, Distribution of data, Class summary
    tab_quality, tab_preprocess, tab_dist, tab_summary = st.tabs([
        "Quality & Features", "Data pipeline and Preprocessing", "Distribution of data", "Class summary"
    ])
    with tab_quality:
        # 1. Key metrics for both data sets
        st.markdown("#### Key metrics")
        raw_sensor_df = load_raw_sensor()
        # features_14.csv
        st.markdown("**features_14 dataset**")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric("Number of rows", f"{len(df):,}")
        with c2:
            with st.container(border=True):
                st.metric("Number of columns", f"{len(df.columns):,}")
        with c3:
            with st.container(border=True):
                n_classes = df["Target"].nunique() if "Target" in df.columns else 0
                st.metric("Number of target classes", n_classes)
        # raw_sensor_data.csv
        st.markdown("**raw_sensor_data dataset**")
        r1, r2, r3, r4 = st.columns(4)
        if raw_sensor_df is not None and not raw_sensor_df.empty:
            task_id_col = next(
                (c for c in raw_sensor_df.columns if str(c).strip().lower() in ("taskid", "taskid_new")),
                raw_sensor_df.columns[0] if len(raw_sensor_df.columns) else None,
            )
            n_tasks = raw_sensor_df[task_id_col].nunique() if task_id_col else 0
            with r1:
                with st.container(border=True):
                    st.metric("Number of rows", f"{len(raw_sensor_df):,}")
            with r2:
                with st.container(border=True):
                    st.metric("Number of rows after aggregation", f"{n_tasks:,}")
            with r3:
                with st.container(border=True):
                    st.metric("Number of columns", f"{len(raw_sensor_df.columns):,}")
            with r4:
                with st.container(border=True):
                    st.metric("Number of tasks (TaskID)", f"{n_tasks:,}")
            
        else:
            with r1:
                with st.container(border=True):
                    st.metric("Number of rows", "—")
            with r2:
                with st.container(border=True):
                    st.metric("Number of columns", "—")
            with r3:
                with st.container(border=True):
                    st.metric("Number of tasks (TaskID)", "—")
            with r4:
                with st.container(border=True):
                    st.metric("Number of rows after aggregation", "—")
            st.caption("raw_sensor_data.csv not found. Add it to the project to see metrics.")

        # 2. Data quality checks table
        st.markdown("#### Data quality checks for features_14 dataset")
        feat_cols = [c for c in df.columns if c != "Target"]
        null_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        const_cols = sum(1 for c in feat_cols if df[c].nunique(dropna=True) <= 1)
        min_class_count = df["Target"].value_counts().min() if "Target" in df.columns else 0

        checks_df = pd.DataFrame({
            "Check": [
                "Total null values",
                "Duplicate rows",
                "Constant / near-constant feature columns",
                "Minimum samples per class",
            ],
            "Value": [
                null_count,
                duplicate_count,
                const_cols,
                min_class_count,
            ],
        })
        st.dataframe(checks_df, use_container_width=True, hide_index=True)

        # 3. Class distribution
        st.markdown("##### Class distribution")
        if "Target" in df.columns:
            fig = maneuver_distribution(df, show_title=False)
            st.plotly_chart(fig, width=700)
        st.markdown("The data is fairly balanced, with turns occurring more often than acceleration or braking.")

    with tab_preprocess:
        st.markdown("#### End-to-End IMU Pipeline")
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<style>.pipeline-diagram table, .pipeline-diagram table td, .pipeline-diagram table th, "
            ".pipeline-diagram tr { border: 0 !important; border-collapse: collapse; background: transparent !important; }</style>",
            unsafe_allow_html=True,
        )
        box = (
            "border: 2px solid #1a6b6b; border-radius: 8px; padding: 16px 24px; "
            "min-width: 180px; min-height: 80px; display: inline-flex; align-items: center; justify-content: center; "
            "font-weight: 600; font-size: 1rem; background: #e8f5f4; color: #1a6b6b; box-sizing: border-box;"
        )
        st.markdown(
            """
            <div class="pipeline-diagram" style="margin:1rem 0;">
            <table style="margin:0 auto; border-collapse:collapse; border:0; background:transparent;">
                <tr>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Raw IMU Sensor Data</div></td>
                <td style="padding:0 4px; border:0 !important; background:transparent;">→</td>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Statistical Feature Aggregation</div></td>
                <td style="padding:0 4px; border:0 !important; background:transparent;">→</td>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Feature Engineering</div></td>
                <td style="padding:0 4px; border:0 !important; background:transparent;">→</td>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Decision Engine</div></td>
                <td style="padding:0 4px; border:0 !important; background:transparent;">→</td>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Alert Output</div></td>
                </tr>
                <tr>
                <td colspan="4" style="padding:0; border:0 !important; background:transparent;"></td>
                <td style="padding:6px; min-width:180px; border:0 !important; background:transparent;"></td>
                <td style="padding:0 4px; border:0 !important; background:transparent;">→</td>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Classification Model</div></td>
                <td style="padding:0 4px; border:0 !important; background:transparent;">→</td>
                <td style="padding:6px; border:0 !important; background:transparent;"><div style="{}">Maneuver prediction</div></td>
                <td colspan="2" style="padding:0; border:0 !important; background:transparent;"></td>
                </tr>
            </table>
            </div>
            """.format(box, box, box, box, box, box, box),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("Raw IMU sensor data is aggregated into statistical features, which are then processed in parallel by a rule-based decision engine and a classification model to generate driving alerts and maneuver predictions.",unsafe_allow_html=True)
        st.markdown("#### About Data")
        st.markdown("This project uses two datasets:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
            <div class="card-title">features_14.csv (Statistical Features)</div>
            <div class="card-text">
                Data is already aggregated by statistical features.Used to design and validate maneuver classification model and a decision engine for alerting.
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
            <div class="card-title">raw_sensor_data.csv (Raw IMU Time-Series)</div>
            <div class="card-text">
                Used to build the pipeline that aggregates raw readings per taskID
                into the same statistical feature format, enabling decision making and prediction.
            </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Why two datasets?")
        st.markdown(
    """
 **features_14.csv** contains a larger number of task-level records with stable,
pre-aggregated statistical features, making it well-suited for training and evaluating the classification model and
decision engine. 

**raw_sensor_data.csv** is used to demonstrate the end-to-end pipeline, showing how raw IMU signals are
converted into statistical features and then passed through the same models to generate maneuver predictions and alerts
for newly collected data.
"""
)

        st.markdown("##### Raw_sensor_data.csv")
        raw_sensor_df = load_raw_sensor()
        if raw_sensor_df is not None and not raw_sensor_df.empty:
            cols_to_drop = [c for c in raw_sensor_df.columns if str(c).startswith("Unnamed")]
            display_df = raw_sensor_df.drop(columns=cols_to_drop, errors="ignore").head()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("##### Features_14.csv")
        if df is not None and not df.empty:
            st.dataframe(df.head(), use_container_width=True, hide_index=True)
        else:
            st.caption("No raw_sensor_data file found (e.g. raw_sensor_data.csv). Add it to the project to see raw IMU data.")
        st.markdown("#### Feature Engineering")
        st.markdown("Turning raw IMU window features into axis score and instability signal, so we can compare behavior without relying on 61 columns.")

        st.markdown("##### Deriving axis scores")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">Intensity</div>
                <div class="card-text">
                    Mean of absolute robust z-scores of magnitude features
                    (mean, median, max, min, sum) per axis, representing how strong the motion is.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">Variability</div>
                <div class="card-text">
                    Mean of robust z-scores of spread features
                    (std, var, cov) per axis, indicating how consistent or unstable the motion is.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <div class="card-title">Impulsiveness</div>
                <div class="card-text">
                    Robust z-score of kurtosis per axis, capturing spikeiness
                    and sudden jerky behavior.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="card">
                <div class="card-title">Directional Bias</div>
                <div class="card-text">
                    Robust z-score of skew per axis, reflecting asymmetry
                    and dominant direction of motion.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Instability Signal Derivation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">Driver Instability</div>
                <div class="card-text">
                Derived from <b>AccX</b>, <b>AccY</b>, and <b>GyroZ</b>, capturing longitudinal and lateral acceleration
                (throttle, braking) and yaw rotation (steering input).
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">Road Instability</div>
                <div class="card-text">
                Primarily derived from <b>AccZ</b> to capture vertical road response. <b>GyroX</b> (roll) and
                <b>GyroY</b> (pitch) are included only when their activity exceeds the median and driver inputs are low,
                isolating road-induced motion.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_dist:
        st.markdown("#### Distribution of data")
        class_val = st.selectbox(
            "Maneuver class",
            options=sorted(df["Target"].dropna().unique()),
            format_func=lambda x: TARGET_LABELS.get(x, str(x)),
            key="data_understanding_class",
        )
        mask = df["Target"] == class_val
        if not mask.any():
            st.warning("No rows for this class.")
        else:
            row = score_df.loc[mask].median()
            fig_hist = axis_distribution_histograms(score_df, mask, AXES, SCORE_COLS)
            if fig_hist.data:
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.caption("No data to plot.")
            st.markdown(
                "Most features across all class types are skewed with long tails, meaning extreme values are present but infrequent. "
                "Since these extremes can distort averages, the median better represents typical behavior than the mean when analyzing class-level data."
            )

    with tab_summary:
        st.markdown("#### Data Confirmation: Do Sensor Signals Reflect Expected Behavior?")
        st.markdown(
                "Yes, The observed sensor patterns closely align with expected vehicle dynamics for each maneuver, "
                "confirming that the data reflects realistic driving behavior and can be reliably used for further prediction and alert decision design. Below is the analysis"
            )
        class_val_summary = st.selectbox(
            "Maneuver class",
            options=sorted(df["Target"].dropna().unique()),
            format_func=lambda x: TARGET_LABELS.get(x, str(x)),
            key="data_understanding_class_summary",
        )
        mask_summary = df["Target"] == class_val_summary
        if not mask_summary.any():
            st.warning("No rows for this class.")
        else:
            row_summary = score_df.loc[mask_summary].median()
            st.caption(f"Values below are the **median** of all windows in this maneuver class.")
            rows = []
            for ax in AXES:
                r = {"Axis": ax}
                for sc in SCORE_COLS:
                    key = f"{ax}_{sc}"
                    r[sc] = row_summary.get(key, None)
                rows.append(r)
            table_df = pd.DataFrame(rows)
            st.dataframe(table_df, use_container_width=True)
            fig = scorecard_grouped_bars(row_summary, AXES, show_title=False)
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            if int(class_val_summary) == 1:
                st.markdown(
                    "For sudden acceleration, forward (AccX) and vertical (AccZ) intensity lead, with relatively consistent lateral and yaw variability and more variable vertical response. Roll (GyroX) shows the most impulsiveness. The pattern fits throttle-in and vehicle pitch or lift rather than steering, with limited lateral and yaw involvement."
                )
            elif int(class_val_summary) == 2:
                st.markdown(
                    "For a sudden right turn, yaw (GyroZ) and lateral (AccY) dominate with strong negative directional bias, consistent with a right turn. AccX adds deceleration and GyroX shows roll to the right. Pitch (GyroY) is more variable, impulsiveness is low, so the turn is sustained rather than spikey."
                )
            elif int(class_val_summary) == 3:
                st.markdown(
                    "For a sudden left turn, roll (GyroX) and yaw (GyroZ) show the highest intensity, followed by lateral (AccY). AccY and GyroZ have strong positive directional bias, consistent with a left turn, while GyroX shows negative bias (roll to the left). AccX stays near zero, impulsiveness is low, so the turn is sustained rather than spikey."
                )
            elif int(class_val_summary) == 4:
                st.markdown(
                    "For sudden braking, forward (AccX) intensity and negative directional bias dominate, with clear vertical (AccZ) response from load transfer. Lateral (AccY) and yaw (GyroZ) stay relatively low. The pattern fits hard braking and pitch rather than steering, with minimal lateral or yaw involvement."
                )


main()
