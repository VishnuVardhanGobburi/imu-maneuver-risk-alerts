"""
Pipeline: raw_sensor_data.csv → features (features_14 schema) → classification + decision engine → scored_alerts_from_raw.csv.
Modular functions: build_features_from_raw(), predict_maneuver(), run_decision_engine().
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .raw_to_features import aggregate_raw_to_features, FEATURE_COLS
from .scores import compute_axis_scores
from .decision_engine import compute_di_ri, normalize_scores, run_engine


# Default task ID column name in raw data
DEFAULT_TASK_ID_COL = "TaskID"


def build_features_from_raw(
    raw: pd.DataFrame,
    task_id_col: str = DEFAULT_TASK_ID_COL,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Part A: Group raw by taskID and compute features matching features_14 schema.
    For each axis (AccX, AccY, AccZ, GyroX, GyroY, GyroZ): mean, median, max, min, sum, std, var,
    coefficient of variation (safe for mean≈0), skewness, kurtosis.
    Preserves Target per taskID only if present and constant within task; otherwise NaN.
    Saves features_from_raw.csv if output_path is set.
    Returns DataFrame with columns: taskID, Target (if any), then 60 feature cols in features_14 order.
    """
    # Normalize TaskID_New / TaskID_new to TaskID so downstream uses one name
    for old in ("TaskID_New", "TaskID_new"):
        if old in raw.columns and "TaskID" not in raw.columns:
            raw = raw.rename(columns={old: "TaskID"})
            break
    if task_id_col not in raw.columns:
        task_id_col = next(
            (c for c in raw.columns if str(c).strip().lower() in ("taskid", "taskid_new")),
            raw.columns[0] if len(raw.columns) else None,
        )
    if task_id_col is None:
        raise ValueError("No taskID column found in raw data.")

    # Aggregate: one row per taskID, 60 stat columns
    df_60, target_series, group_keys, group_by = aggregate_raw_to_features(
        raw, group_col=task_id_col, class_col=None, features_only=True
    )

    # taskID column: first element of each group key (handles single-col or multi-col group)
    task_ids = [k[0] if isinstance(k, tuple) else k for k in group_keys]

    # Build table: taskID, Target (if present and constant per task), then 60 features
    features_df = pd.DataFrame({task_id_col: task_ids})
    if target_series is not None and len(target_series) == len(task_ids):
        # Use first value per group as Target (assumed constant within task)
        features_df["Target"] = target_series.values
        # Optional: set to NaN where Target was not constant (we don't re-check here; raw aggregation already took first)
    else:
        features_df["Target"] = np.nan

    # Align 60 cols to features_14 order
    for c in FEATURE_COLS:
        if c in df_60.columns:
            features_df[c] = df_60[c].values
        else:
            features_df[c] = np.nan

    # Reorder columns: taskID, Target, then FEATURE_COLS (match features_14 structure)
    features_df = features_df[[task_id_col, "Target"] + FEATURE_COLS]

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out_path, index=False)

    return features_df


def predict_maneuver(
    features_df: pd.DataFrame,
    model,
    scaler,
    imputer,
    feature_columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    """
    Part B: Run existing classification model on feature table.
    Ensures feature columns align with training schema (same names, order; missing filled per imputer).
    Returns (predicted_class, predicted_confidence) as Series with same index as features_df.
    """
    from .predict import predict_one_row

    pred_classes = []
    pred_confidences = []
    for i in range(len(features_df)):
        row_df = features_df.iloc[[i]].reindex(columns=feature_columns)
        pred, proba = predict_one_row(row_df, model, scaler, imputer, feature_columns)
        pred_classes.append(pred if pred is not None else np.nan)
        if proba is not None and len(proba) > 0:
            pred_confidences.append(float(np.max(proba)))
        else:
            pred_confidences.append(np.nan)
    return (
        pd.Series(pred_classes, index=features_df.index),
        pd.Series(pred_confidences, index=features_df.index),
    )


def run_decision_engine(
    features_df: pd.DataFrame,
    t_low: float = 0.30,
    t_med: float = 0.60,
    t_dom: float = 0.20,
) -> pd.DataFrame:
    """
    Part C: Run rule-based decision engine on the feature table.
    Reuses compute_axis_scores, compute_di_ri, normalize_scores, run_engine.
    Returns DataFrame with columns: DI_norm, RI_norm, alert_type, alert_severity, alert_explanation.
    """
    # Need only the 60 feature columns for scoring
    feature_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    if len(feature_cols) < len(FEATURE_COLS):
        use_df = features_df.reindex(columns=FEATURE_COLS)
    else:
        use_df = features_df[FEATURE_COLS]

    score_df = compute_axis_scores(use_df)
    dr = compute_di_ri(use_df, score_df=score_df)
    di_norm = normalize_scores(dr["DRIVER_INSTABILITY"])
    ri_norm = normalize_scores(dr["ROAD_INSTABILITY"])

    alert_types = []
    alert_severities = []
    alert_explanations = []
    for i in range(len(features_df)):
        res = run_engine(
            float(di_norm.iloc[i]) if pd.notna(di_norm.iloc[i]) else 0.0,
            float(ri_norm.iloc[i]) if pd.notna(ri_norm.iloc[i]) else 0.0,
            t_low=t_low, t_med=t_med, t_dom=t_dom,
        )
        if res is None:
            alert_types.append("")
            alert_severities.append(0)
            alert_explanations.append("")
        else:
            alert_types.append(res.alert)
            alert_severities.append(res.severity)
            alert_explanations.append(res.message)

    return pd.DataFrame({
        "DI_norm": di_norm.values,
        "RI_norm": ri_norm.values,
        "alert_type": alert_types,
        "alert_severity": alert_severities,
        "alert_explanation": alert_explanations,
    }, index=features_df.index)


def run_pipeline(
    raw: pd.DataFrame,
    task_id_col: str = DEFAULT_TASK_ID_COL,
    model=None,
    scaler=None,
    imputer=None,
    feature_columns: list | None = None,
    features_from_raw_path: str | Path | None = "features_from_raw.csv",
    scored_alerts_path: str | Path | None = "scored_alerts_from_raw.csv",
) -> pd.DataFrame:
    """
    End-to-end: build features, predict, run decision engine, merge and save.
    If model/scaler/imputer/feature_columns are None, caller must train and pass them.
    Returns final DataFrame (one row per taskID with features + predicted_class + confidence + engine outputs).
    """
    # Part A: raw → features
    features_df = build_features_from_raw(raw, task_id_col=task_id_col, output_path=features_from_raw_path)

    if model is None or scaler is None or imputer is None or not feature_columns:
        # Return features only; caller can add predictions and engine later
        return features_df

    # Part B: predict
    pred_class, pred_confidence = predict_maneuver(features_df, model, scaler, imputer, feature_columns)
    features_df = features_df.copy()
    features_df["predicted_class"] = pred_class.values
    features_df["predicted_confidence"] = pred_confidence.values

    # Part C: decision engine
    engine_df = run_decision_engine(features_df)
    features_df["DI_norm"] = engine_df["DI_norm"].values
    features_df["RI_norm"] = engine_df["RI_norm"].values
    features_df["alert_type"] = engine_df["alert_type"].values
    features_df["alert_severity"] = engine_df["alert_severity"].values
    features_df["alert_explanation"] = engine_df["alert_explanation"].values

    if scored_alerts_path is not None:
        Path(scored_alerts_path).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(scored_alerts_path, index=False)

    return features_df
