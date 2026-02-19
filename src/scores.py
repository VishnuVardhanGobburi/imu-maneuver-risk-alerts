"""
Robust z-score, axis intelligence scores, composite indices, class-level summary.
"""

import numpy as np
import pandas as pd
from .parse import AXES, columns_for_axis_score


def robust_z(series: pd.Series) -> pd.Series:
    """z_f = (f - median) / IQR; if IQR==0 then z=0."""
    med = series.median()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(0.0, index=series.index)
    return (series - med) / iqr


def compute_axis_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row scores for each axis: Intensity, Variability, Impulsiveness, Directional_Bias.
    Composite: DRIVER_INSTABILITY, ROAD_INSTABILITY.
    """
    result = pd.DataFrame(index=df.index)
    for axis in AXES:
        for score_type, key_suffix in [
            ("INTENSITY", "Intensity"),
            ("VARIABILITY", "Variability"),
            ("IMPULSIVENESS", "Impulsiveness"),
            ("DIRECTIONAL_BIAS", "Directional_Bias"),
        ]:
            cols = columns_for_axis_score(df, axis, score_type)
            if not cols:
                result[f"{axis}_{key_suffix}"] = np.nan
                continue
            z_list = [robust_z(df[c]) for c in cols]
            if score_type == "INTENSITY":
                result[f"{axis}_{key_suffix}"] = pd.concat([z.abs() for z in z_list], axis=1).mean(axis=1)
            elif score_type == "VARIABILITY":
                result[f"{axis}_{key_suffix}"] = pd.concat(z_list, axis=1).mean(axis=1)
            elif score_type in ("IMPULSIVENESS", "DIRECTIONAL_BIAS"):
                result[f"{axis}_{key_suffix}"] = z_list[0]

    # Composites (use available components; NaN-safe mean)
    v_ax = result.get("AccX_Variability", pd.Series(0.0, index=df.index)).fillna(0)
    v_ay = result.get("AccY_Variability", pd.Series(0.0, index=df.index)).fillna(0)
    v_gz = result.get("GyroZ_Variability", pd.Series(0.0, index=df.index)).fillna(0)
    i_ax = result.get("AccX_Impulsiveness", pd.Series(0.0, index=df.index)).fillna(0)
    i_ay = result.get("AccY_Impulsiveness", pd.Series(0.0, index=df.index)).fillna(0)
    i_gz = result.get("GyroZ_Impulsiveness", pd.Series(0.0, index=df.index)).fillna(0)
    result["DRIVER_INSTABILITY"] = (v_ax + v_ay + v_gz) / 3 + (i_ax + i_ay + i_gz) / 3

    v_az = result.get("AccZ_Variability", pd.Series(0.0, index=df.index)).fillna(0)
    i_az = result.get("AccZ_Impulsiveness", pd.Series(0.0, index=df.index)).fillna(0)
    base_ri = v_az + i_az

    # Conditional road terms: GyroX when steering is low, GyroY when throttle/braking is low (15th percentile)
    p15 = 0.15
    steering_strength = result.get("GyroZ_Intensity", pd.Series(np.nan, index=df.index))
    long_strength = result.get("AccX_Intensity", pd.Series(np.nan, index=df.index))
    p15_steering = steering_strength.quantile(p15)
    p15_long = long_strength.quantile(p15)
    low_steering = steering_strength <= p15_steering
    low_longitudinal = long_strength <= p15_long

    v_gx = result.get("GyroX_Variability", pd.Series(0.0, index=df.index)).fillna(0)
    i_gx = result.get("GyroX_Impulsiveness", pd.Series(0.0, index=df.index)).fillna(0)
    v_gy = result.get("GyroY_Variability", pd.Series(0.0, index=df.index)).fillna(0)
    i_gy = result.get("GyroY_Impulsiveness", pd.Series(0.0, index=df.index)).fillna(0)
    gyro_x_activity = v_gx + i_gx
    gyro_y_activity = v_gy + i_gy
    # Only add GyroX/GyroY when their activity is above median (meaningful roll/pitch)
    high_gyro_x = gyro_x_activity >= gyro_x_activity.quantile(0.5)
    high_gyro_y = gyro_y_activity >= gyro_y_activity.quantile(0.5)
    add_gyro_x = low_steering & high_gyro_x
    add_gyro_y = low_longitudinal & high_gyro_y
    ri_gyro_x = (v_gx + i_gx) * add_gyro_x.astype(float)
    ri_gyro_y = (v_gy + i_gy) * add_gyro_y.astype(float)
    # Weight 0.5 so AccZ remains dominant; conditional terms only refine
    result["ROAD_INSTABILITY"] = base_ri + 0.5 * ri_gyro_x + 0.5 * ri_gyro_y

    return result


def class_summary(df: pd.DataFrame, score_df: pd.DataFrame, target_col: str = "Target", bootstrap_n: int = 200) -> pd.DataFrame:
    """Per (class, axis, score_type): mean, lower, upper (bootstrap CI)."""
    merged = score_df.copy()
    merged["Target"] = df[target_col].values
    rows = []
    for cls in sorted(merged["Target"].dropna().unique()):
        sub = merged[merged["Target"] == cls]
        for col in score_df.columns:
            if "_" not in col:
                continue
            if col in ("DRIVER_INSTABILITY", "ROAD_INSTABILITY"):
                continue
            axis = col.rsplit("_", 1)[0]
            score_type = col.rsplit("_", 1)[1]
            vals = sub[col].dropna()
            if len(vals) == 0:
                rows.append({"class": cls, "axis": axis, "score_type": score_type, "mean": np.nan, "lower": np.nan, "upper": np.nan})
                continue
            mean_val = vals.mean()
            if len(vals) < 2:
                rows.append({"class": cls, "axis": axis, "score_type": score_type, "mean": mean_val, "lower": mean_val, "upper": mean_val})
                continue
            boot = np.array([vals.sample(n=len(vals), replace=True).mean() for _ in range(bootstrap_n)])
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            rows.append({"class": cls, "axis": axis, "score_type": score_type, "mean": mean_val, "lower": lo, "upper": hi})
    return pd.DataFrame(rows)


def quality_checks(df: pd.DataFrame, score_df: pd.DataFrame) -> dict:
    """Missing axis coverage, % NaN per score type, Target class count."""
    axes_covered = {}
    for ax in AXES:
        cols = [c for c in score_df.columns if c.startswith(ax + "_")]
        axes_covered[ax] = len(cols)
    nan_pct = score_df.isna().mean() * 100
    target_classes = df["Target"].nunique() if "Target" in df.columns else 0
    return {
        "axes_covered": axes_covered,
        "nan_pct": nan_pct,
        "target_n_classes": target_classes,
        "target_ok": target_classes == 4,
    }
