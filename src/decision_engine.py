"""
Decision Engine: DI/RI computation, normalization to [0,1], and alert logic.
Level 0 (both low): returns None — no alert message or UI output.
Levels 1–3: returns a structured alert (alert_type, severity, cause, message, explanation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import NamedTuple

# Reuse app scoring if available to avoid duplication
try:
    from .scores import compute_axis_scores
except ImportError:
    compute_axis_scores = None


def compute_di_ri(df: pd.DataFrame, score_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return a dataframe with DRIVER_INSTABILITY and ROAD_INSTABILITY (DI, RI).
    If score_df is provided and has these columns, use them; else compute from df.
    """
    out = pd.DataFrame(index=df.index)
    if score_df is not None and "DRIVER_INSTABILITY" in score_df.columns and "ROAD_INSTABILITY" in score_df.columns:
        out["DRIVER_INSTABILITY"] = score_df["DRIVER_INSTABILITY"].values
        out["ROAD_INSTABILITY"] = score_df["ROAD_INSTABILITY"].values
        return out
    if compute_axis_scores is None:
        out["DRIVER_INSTABILITY"] = np.nan
        out["ROAD_INSTABILITY"] = np.nan
        return out
    scores = compute_axis_scores(df)
    out["DRIVER_INSTABILITY"] = scores["DRIVER_INSTABILITY"].values
    out["ROAD_INSTABILITY"] = scores["ROAD_INSTABILITY"].values
    return out


def normalize_scores(series: pd.Series) -> pd.Series:
    """
    Scale to [0, 1] using percentile (rank) scaling. Robust to outliers.
    """
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=series.index)
    return series.rank(pct=True, method="average").clip(0.0, 1.0)


class EngineResult(NamedTuple):
    """Structured alert for Levels 1–3. Level 0 returns None (no alert)."""
    cause: str
    alert: str          # alert_type (e.g. "Driver Warning")
    severity: int       # 1, 2, or 3
    message: str        # custom human-readable alert message
    explanation: str    # brief reason why the alert was triggered


def run_engine(
    di_norm: float,
    ri_norm: float,
    t_low: float = 0.30,
    t_med: float = 0.60,
    t_dom: float = 0.20,
) -> EngineResult | None:
    """
    Decision logic: first match wins.
    Level 0 (DI < 0.30 and RI < 0.30): returns None — no alert.
    Levels 1–3: returns EngineResult with alert_type, severity, cause, message, explanation.
    """
    di_norm = float(di_norm) if pd.notna(di_norm) else 0.0
    ri_norm = float(ri_norm) if pd.notna(ri_norm) else 0.0

    # ——— Level 0: no alert ———
    if di_norm < t_low and ri_norm < t_low:
        return None

    # ——— Mixed – Level 3 ———
    if di_norm >= t_med and ri_norm >= t_med:
        return EngineResult(
            cause="Mixed",
            alert="Safety Alert",
            severity=3,
            message="High-risk situation: Reduce speed and avoid sudden steering or braking until road conditions improve.",
            explanation="Both driver and road instability are high.",
        )

    # ——— Driver-dominant – Level 3 ———
    if (di_norm - ri_norm) > t_dom and di_norm >= 0.80:
        return EngineResult(
            cause="Driver-dominant",
            alert="Driver Warning",
            severity=3,
            message="Highly aggressive driving detected. Immediate correction recommended.",
            explanation="Driver instability is very high and dominates road contribution.",
        )

    # ——— Driver-dominant – Level 2 ———
    if (di_norm > 0.75 and ri_norm < t_low) or (
        (di_norm - ri_norm) > t_dom and t_med <= di_norm < 0.80
    ) or (di_norm >= ri_norm and di_norm >= t_med):
        return EngineResult(
            cause="Driver-dominant",
            alert="Driver Warning",
            severity=2,
            message="Aggressive driving behavior detected. Reduce speed and steering intensity.",
            explanation="Driver instability is elevated.",
        )

    # ——— Road-dominant – Level 2 ———
    if ri_norm > 0.75 and di_norm < t_low:
        return EngineResult(
            cause="Road-dominant",
            alert="Road Advisory",
            severity=2,
            message="Severe road irregularities detected. Expect reduced vehicle comfort and control.",
            explanation="Road instability is high and driver contribution is low.",
        )

    # ——— Road-dominant – Level 1 ———
    if (ri_norm - di_norm) > t_dom and ri_norm >= t_med:
        return EngineResult(
            cause="Road-dominant",
            alert="Road Advisory",
            severity=1,
            message="Road conditions are disturbing vehicle stability. Drive cautiously.",
            explanation="Road instability dominates; context may explain the event.",
        )
    if ri_norm > di_norm:
        return EngineResult(
            cause="Road-dominant",
            alert="Road Advisory",
            severity=1,
            message="Road conditions are disturbing vehicle stability. Drive cautiously.",
            explanation="Road contribution exceeds driver contribution.",
        )

    # ——— Driver-dominant – Level 1 ———
    # DI ≥ RI and DI < 0.60
    return EngineResult(
        cause="Driver-dominant",
        alert="Driver Warning",
        severity=1,
        message="Driving inputs are slightly aggressive. Maintain smoother control.",
        explanation="Driver is the primary contributor but below high threshold.",
    )
