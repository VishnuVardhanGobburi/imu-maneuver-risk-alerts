"""
Feature family detection: sensor (Acc/Gyro), axis (X/Y/Z), metric tokens.
Column names are source of truth.
"""

import re
from typing import NamedTuple

AXES = ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ")

# Tokens for score components (case-insensitive)
INTENSITY_TOKENS = ("mean", "median", "max", "min", "sum")
VARIABILITY_TOKENS = ("std", "stddev", "var", "cov")
IMPULSE_TOKENS = ("kurt",)
BIAS_TOKENS = ("skew",)


class AxisMetric(NamedTuple):
    axis: str  # AccX, AccY, AccZ, GyroX, GyroY, GyroZ
    token: str  # mean, std, kurt, etc.


def get_axis(col: str) -> str | None:
    """Return AccX, AccY, AccZ, GyroX, GyroY, GyroZ or None. Suffix X/Y/Z is source of truth."""
    if not isinstance(col, str) or len(col) < 3:
        return None
    col_lower = col.lower()
    sensor = None
    if col_lower.startswith("acc") and "gyro" not in col_lower[:5]:
        sensor = "Acc"
    elif col_lower.startswith("gyro"):
        sensor = "Gyro"
    if sensor is None:
        return None
    last = col[-1].upper()
    if last in ("X", "Y", "Z"):
        return sensor + last
    return None


def get_token(col: str, tokens: tuple) -> str | None:
    """Return first matching token (case-insensitive) or None."""
    col_lower = col.lower()
    for t in tokens:
        if t in col_lower:
            return t
    return None


def columns_for_axis_score(df, axis: str, score_type: str) -> list[str]:
    """
    score_type: INTENSITY, VARIABILITY, IMPULSIVENESS, DIRECTIONAL_BIAS.
    Return list of column names belonging to axis with matching token.
    """
    out = []
    for c in df.columns:
        if c == "Target":
            continue
        if get_axis(c) != axis:
            continue
        if score_type == "INTENSITY":
            if get_token(c, INTENSITY_TOKENS):
                out.append(c)
        elif score_type == "VARIABILITY":
            if get_token(c, VARIABILITY_TOKENS):
                out.append(c)
        elif score_type == "IMPULSIVENESS":
            if get_token(c, IMPULSE_TOKENS):
                out.append(c)
        elif score_type == "DIRECTIONAL_BIAS":
            if get_token(c, BIAS_TOKENS):
                out.append(c)
    return out
