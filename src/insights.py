"""
Rule-based insight generator from axis scores. Uses percentiles for justification.
"""

import numpy as np
import pandas as pd


def _pct_label(series: pd.Series, value: float) -> str:
    """e.g. top 10% or bottom 20%."""
    if series.isna().all() or pd.isna(value):
        return ""
    clean = series.dropna()
    if len(clean) == 0:
        return ""
    pct = (clean < value).mean() * 100
    if pct >= 90:
        return "top 10%"
    if pct >= 75:
        return "top 25%"
    if pct >= 50:
        return "above median"
    if pct >= 25:
        return "below median"
    if pct >= 10:
        return "bottom 25%"
    return "bottom 10%"


def generate_insights(
    row: pd.Series,
    score_df: pd.DataFrame,
    driver_road: pd.DataFrame | None = None,
    n: int = 8,
) -> list[tuple[str, str]]:
    """
    Return list of (insight_text, justification). Uses thresholds from dataset percentiles.
    """
    insights = []
    if score_df is None or score_df.empty:
        return insights

    def get(name: str, default: float = 0.0):
        v = row.get(name, default)
        return float(v) if pd.notna(v) else default

    # Percentile thresholds (from full score_df)
    def pct90(name):
        s = score_df[name] if name in score_df.columns else pd.Series([0])
        return s.quantile(0.9) if len(s.dropna()) else 0

    def pct10(name):
        s = score_df[name] if name in score_df.columns else pd.Series([0])
        return s.quantile(0.1) if len(s.dropna()) else 0

    # AccX intensity + bias
    i_ax = get("AccX_Intensity")
    b_ax = get("AccX_Directional_Bias")
    if i_ax >= pct90("AccX_Intensity") and b_ax > 0:
        j = _pct_label(score_df["AccX_Intensity"], i_ax) or "high"
        insights.append(("Acceleration-dominant longitudinal motion.", f"because AccX intensity is in the {j}"))
    if i_ax >= pct90("AccX_Intensity") and b_ax < 0:
        j = _pct_label(score_df["AccX_Intensity"], i_ax) or "high"
        insights.append(("Braking-dominant longitudinal motion.", f"because AccX intensity is in the {j} and bias is negative"))

    # AccZ vertical
    v_az = get("AccZ_Variability")
    i_az = get("AccZ_Impulsiveness")
    if v_az >= pct90("AccZ_Variability") or i_az >= pct90("AccZ_Impulsiveness"):
        insights.append(("Vertical instability suggests roughness/bumps.", "because AccZ variability or impulsiveness is elevated"))

    # GyroZ
    v_gz = get("GyroZ_Variability")
    if v_gz >= pct90("GyroZ_Variability"):
        insights.append(("Steering/yaw instability is elevated.", "because GyroZ variability is in the top range"))

    # Driver vs road (from composite columns if passed)
    if driver_road is not None and "DRIVER_INSTABILITY" in row.index:
        di = get("DRIVER_INSTABILITY")
        ri = get("ROAD_INSTABILITY")
        if "DRIVER_INSTABILITY" in driver_road.columns:
            di_90 = driver_road["DRIVER_INSTABILITY"].quantile(0.9)
            ri_90 = driver_road["ROAD_INSTABILITY"].quantile(0.9)
            if di >= di_90 and ri < ri_90:
                insights.append(("Driver-driven aggressiveness dominates.", "because driver instability is high and road instability is lower"))
            if ri >= ri_90 and di < di_90:
                insights.append(("Road-driven roughness dominates.", "because road instability is high and driver instability is lower"))

    # Bias GyroZ / AccY
    b_gz = get("GyroZ_Directional_Bias")
    b_ay = get("AccY_Directional_Bias")
    if abs(b_gz) >= 0.5:
        insights.append(("Yaw bias indicates turn direction.", "from GyroZ directional bias sign and magnitude"))
    if abs(b_ay) >= 0.5:
        insights.append(("Lateral bias indicates lateral force direction.", "from AccY directional bias"))

    return insights[:n]


def executive_summary_lines(score_df: pd.DataFrame, class_summary_df: pd.DataFrame, target_col_series: pd.Series) -> list[str]:
    """Top 6 insights across classes for executive_insights.md."""
    lines = []
    # By class: which has highest driver instability mean
    if "DRIVER_INSTABILITY" in score_df.columns:
        by_class = score_df.copy()
        by_class["Target"] = target_col_series.values
        means = by_class.groupby("Target")["DRIVER_INSTABILITY"].mean()
        if len(means) > 0:
            top_class = means.idxmax()
            lines.append(f"Class {top_class} has the highest mean driver instability in this dataset.")
    # AccZ roughness
    if "AccZ_Variability" in score_df.columns:
        lines.append("AccZ variability and impulsiveness are associated with vertical/road roughness.")
    # GyroZ turning
    if "GyroZ_Variability" in score_df.columns:
        lines.append("GyroZ (yaw) variability tends to dominate turning-related events.")
    lines.append("Axis Intelligence scores are robust z-scores; interpret relative to the full sample.")
    lines.append("Findings are associative; they do not imply causality.")
    return lines[:6]
