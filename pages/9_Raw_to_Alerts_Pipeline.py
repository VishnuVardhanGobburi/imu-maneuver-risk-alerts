"""
Page 9: Raw-to-Alerts Pipeline.
Reads raw_sensor_data.csv → aggregates to features_14-style features per taskID →
runs classification model + decision engine → saves features_from_raw.csv and scored_alerts_from_raw.csv.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from src.io import ensure_data, load_raw_sensor
from src.layout import inject_full_width
from src.pipeline import (
    build_features_from_raw,
    predict_maneuver,
    run_decision_engine,
    DEFAULT_TASK_ID_COL,
)
from src.predict import train_and_evaluate
from src.viz import TARGET_LABELS
from src.viz_decision import build_gauges_or_bars, build_interaction_map
from src.decision_engine import run_engine, compute_di_ri
from src.scores import compute_axis_scores


_BASE = Path(__file__).resolve().parent.parent


@st.cache_data
def _cached_train(df: pd.DataFrame, _cache_version: int = 5):
    """Train classifier on main feature data. Bump _cache_version to force retrain after model changes."""
    return train_and_evaluate(
        df,
        test_size=0.25,
        random_state=42,
        C=1.0,
        max_iter=1000,
        use_ordered_split=False,
    )


def main():
    st.set_page_config(
        page_title="From Sensors to Alerts & Predictions",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_full_width()
    st.markdown("## From Raw-to-Alerts Pipeline")
    st.markdown("This page converts raw sensor readings into task-level statistical features, runs maneuver prediction and the decision engine, and allows to select any raw record to view both the original sensor data and the complete pipeline output for its TaskID.")
    # Resolve output paths (project root)
    features_path = _BASE / "features_from_raw.csv"
    scored_path = _BASE / "scored_alerts_from_raw.csv"

    # Load raw data
    raw = load_raw_sensor()
    if raw is None:
        st.warning(
            "**raw_sensor_data.csv** not found (looked in project root and /mnt/data). "
            "Add raw_sensor_data.csv with column **TaskID** (or TaskID_new, renamed to TaskID) and sensor columns: AccX, AccY, AccZ, GyroX, GyroY, GyroZ."
        )
        return

    # Detect taskID column
    task_id_col = (
        DEFAULT_TASK_ID_COL
        if DEFAULT_TASK_ID_COL in raw.columns
        else next(
            (c for c in raw.columns if str(c).strip().lower() == "taskid_new"),
            raw.columns[0],
        )
    )

    try:
        features_df = build_features_from_raw(
            raw, task_id_col=task_id_col, output_path=str(features_path)
        )
    except Exception as e:
        st.error(f"Feature build failed: {e}")
        return

    # Train model on main data (for prediction)
    df_main, *_ = ensure_data()
    result = _cached_train(df_main, _cache_version=5)
    if result.get("error"):
        st.error(result["error"])
        return
    model = result["model"]
    scaler = result["scaler"]
    imputer = result["imputer"]
    feature_columns = result["feature_columns"]

    pred_class, pred_confidence = predict_maneuver(
        features_df, model, scaler, imputer, feature_columns
    )
    features_df = features_df.copy()
    features_df["predicted_class"] = pred_class.values
    features_df["predicted_confidence"] = pred_confidence.values

    engine_df = run_decision_engine(features_df)
    features_df["DI_norm"] = engine_df["DI_norm"].values
    features_df["RI_norm"] = engine_df["RI_norm"].values
    features_df["alert_type"] = engine_df["alert_type"].values
    features_df["alert_severity"] = engine_df["alert_severity"].values
    features_df["alert_explanation"] = engine_df["alert_explanation"].values

    features_df.to_csv(scored_path, index=False)
#    st.divider()
    # ---- Select a row from raw_sensor_data.csv (before conversion) ----
    n_raw = len(raw)
    raw_row_options = list(range(n_raw))
    st.markdown("#### Select a raw record to see its pipeline output")
    raw_sel = st.selectbox(
        "Select or type a row number",
        raw_row_options,
        index=0,
        format_func=lambda i: f"Row {i}",
        key="pipeline_raw_row_sel",
    )
    cols_to_show = [c for c in raw.columns if not str(c).startswith("Unnamed")]
    selected_raw_row = raw.iloc[raw_sel: raw_sel + 1][cols_to_show]
    st.dataframe(selected_raw_row, use_container_width=True, hide_index=True)

    # ---- Pipeline output for this TaskID ----
    task_id = raw[task_id_col].iloc[raw_sel]
    task_rows = features_df[features_df[task_id_col] == task_id]
    if task_rows.empty:
        st.warning(f"No pipeline output for TaskID {task_id}. This task may have been dropped during feature building.")
    else:
        task_iloc = features_df.index.get_loc(task_rows.index[0])

        di_val = float(features_df["DI_norm"].iloc[task_iloc])
        ri_val = float(features_df["RI_norm"].iloc[task_iloc])
        res = run_engine(di_val, ri_val, t_low=0.30, t_med=0.60, t_dom=0.20)

        score_df = compute_axis_scores(features_df)
        dr = compute_di_ri(features_df, score_df=score_df)
        di_raw = float(dr["DRIVER_INSTABILITY"].iloc[task_iloc])
        ri_raw = float(dr["ROAD_INSTABILITY"].iloc[task_iloc])
        pred = features_df["predicted_class"].iloc[task_iloc]
        pred_label = TARGET_LABELS.get(int(pred), str(pred)) if pd.notna(pred) else "—"

        # Charts first (output for this TaskID); message color by severity (green=no alert, yellow=1, orange=2, red=3)
        st.markdown("#### Pipeline output for this TaskID")
        _msg_colors = {0: "#2e7d32", 1: "#b8860b", 2: "#ef6c00", 3: "#c62828"}
        if res is not None:
            color = _msg_colors.get(res.severity, "#333")
            st.markdown(f'<p style="color: {color}; font-weight: bold;">Message: {res.message}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #2e7d32; font-weight: 600;">**No alerts Needed**</p>', unsafe_allow_html=True)
        fig_bars = build_gauges_or_bars(di_val, ri_val, title=f"TaskID {task_id}: DI_Norm & RI_Norm")
        st.plotly_chart(fig_bars, use_container_width=True)

        ri_norm_series = features_df["RI_norm"]
        di_norm_series = features_df["DI_norm"]
        # Use Target if available, else predicted_class so the chart shows 4 colors by maneuver
        target_series = (
            features_df["Target"]
            if "Target" in features_df.columns and features_df["Target"].notna().any()
            else features_df["predicted_class"]
        )
        fig_map = build_interaction_map(
            ri_norm_series,
            di_norm_series,
            selected_idx=task_iloc,
            t_med=0.60,
            target=target_series,
            target_labels=TARGET_LABELS,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        # Current event summary at the end (Level 0: no alert — empty Cause/Alert/Severity)
        row_out = pd.Series({
            "TaskID": task_id,
            "DI": round(di_raw, 4),
            "RI": round(ri_raw, 4),
            "DI_Norm": round(di_val, 4),
            "RI_Norm": round(ri_val, 4),
            "Cause": res.cause if res is not None else "",
            "Alert": res.alert if res is not None else "",
            "Severity": res.severity if res is not None else 0,
            "Predicted class": pred_label,
        })
        st.markdown("#### Current event summary")
        st.dataframe(row_out.to_frame().T, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
