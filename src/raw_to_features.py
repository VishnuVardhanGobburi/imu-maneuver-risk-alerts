"""
Build the 60 statistical feature columns from raw sensor data by grouping by TaskID.
For each TaskID (e.g. 1), compute mean and other stats over all records in that group for each sensor column.
Output: one row per TaskID with only the 60 feature cols (same as features_14.csv, no class column).
"""

import numpy as np
import pandas as pd

# Expected feature column order (60 columns, matching features_14.csv)
FEATURE_COLS = [
    "AccMeanX", "AccMeanY", "AccMeanZ", "AccCovX", "AccCovY", "AccCovZ",
    "AccSkewX", "AccSkewY", "AccSkewZ", "AccKurtX", "AccKurtY", "AccKurtZ",
    "AccSumX", "AccSumY", "AccSumZ", "AccMinX", "AccMinY", "AccMinZ",
    "AccMaxX", "AccMaxY", "AccMaxZ", "AccVarX", "AccVarY", "AccVarZ",
    "AccMedianX", "AccMedianY", "AccMedianZ", "AccStdX", "AccStdY", "AccStdZ",
    "GyroMeanX", "GyroMeanY", "GyroMeanZ", "GyroCovX", "GyroCovY", "GyroCovZ",
    "GyroSkewX", "GyroSkewY", "GyroSkewZ", "GyroSumX", "GyroSumY", "GyroSumZ",
    "GyroKurtX", "GyroKurtY", "GyroKurtZ", "GyroMinX", "GyroMinY", "GyroMinZ",
    "GyroMaxX", "GyroMaxY", "GyroMaxZ", "GyroVarX", "GyroVarY", "GyroVarZ",
    "GyroMedianX", "GyroMedianY", "GyroMedianZ", "GyroStdX", "GyroStdY", "GyroStdZ",
]

# Map (sensor, axis) to raw column name variants we accept
def _raw_col(raw: pd.DataFrame, sensor: str, axis: str) -> str | None:
    """Return first matching column for sensor+axis (e.g. AccX, acc_x, Acc_X)."""
    candidates = [
        f"{sensor}{axis}",
        f"{sensor}_{axis}",
        f"{sensor} {axis}",
        f"{sensor.lower()}{axis.lower()}",
        f"{sensor.lower()}_{axis.lower()}",
    ]
    for c in candidates:
        if c in raw.columns:
            return c
    return None


def _cov(series: pd.Series) -> float:
    """Coefficient of variation; 0 if mean is 0 or series too short."""
    if series is None or len(series) < 2:
        return np.nan
    m = series.mean()
    if m == 0 or np.isnan(m):
        return 0.0
    return float(series.std() / m)


def aggregate_raw_to_features(
    raw: pd.DataFrame,
    group_col: str = "TaskID",
    class_col: str | None = None,
    sensor_cols: dict | None = None,
    features_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Group raw sensor data by group_col (e.g. TaskID); compute 60 stats per group.
    E.g. for TaskID=1: mean(AccX) over all records with TaskID=1, and likewise for all stats and all columns.
    raw: DataFrame with TaskID and sensor columns (AccX, AccY, AccZ, GyroX, GyroY, GyroZ).
    group_col: column that identifies the task/window (default TaskID).
    class_col: optional; if set, group by (group_col, class_col) and return target_series for "Actual".
    sensor_cols: optional dict mapping (sensor, axis) to column name.
    features_only: if True, return df with only the 60 FEATURE_COLS (no class column).
    Returns (df, target_series, group_keys): df has 60 cols only; target_series is class per row or None;
    group_keys is a list of the group key(s) for each row (e.g. TaskID value), for looking up original records.
    """
    if group_col not in raw.columns:
        raise ValueError(f"Group column '{group_col}' not found in raw data. Columns: {list(raw.columns)}")

    if class_col and class_col in raw.columns:
        group_by = [group_col, class_col]
    else:
        group_by = [group_col]
        class_col = None

    # Resolve sensor columns
    if sensor_cols is None:
        sensor_cols = {}
    axes = [("Acc", "X"), ("Acc", "Y"), ("Acc", "Z"), ("Gyro", "X"), ("Gyro", "Y"), ("Gyro", "Z")]
    col_map = {}
    for s, a in axes:
        key = (s, a)
        if key in sensor_cols:
            if sensor_cols[key] in raw.columns:
                col_map[key] = sensor_cols[key]
            continue
        c = _raw_col(raw, s, a)
        if c is None:
            raise ValueError(f"No column found for {s}{a}. Available: {list(raw.columns)}")
        col_map[key] = c

    # Stats to compute per axis (name_suffix, agg)
    def mean_(s): return s.mean()
    def cov_(s): return _cov(s)
    def skew_(s): return s.skew() if len(s) >= 3 else np.nan
    def kurt_(s): return s.kurtosis() if len(s) >= 4 else np.nan
    def sum_(s): return s.sum()
    def min_(s): return s.min()
    def max_(s): return s.max()
    def var_(s): return s.var()
    def median_(s): return s.median()
    def std_(s): return s.std()

    stat_specs = [
        ("Mean", mean_), ("Cov", cov_), ("Skew", skew_), ("Kurt", kurt_),
        ("Sum", sum_), ("Min", min_), ("Max", max_), ("Var", var_),
        ("Median", median_), ("Std", std_),
    ]

    rows = []
    targets = []
    group_keys = []
    # If raw has a class/target column, capture it per group (for "Actual" on validation) even when not grouping by it
    label_col = next((c for c in ["Target", "Class", "target", "class"] if c in raw.columns), None)

    for keys, group in raw.groupby(group_by, sort=False):
        keys = (keys,) if len(group_by) == 1 else keys
        group_keys.append(keys[0] if len(keys) == 1 else keys)
        if class_col:
            targets.append(group[class_col].iloc[0])
        elif label_col:
            targets.append(group[label_col].iloc[0])
        row = {}
        for (sensor, axis), col in col_map.items():
            s = group[col].dropna()
            for stat_name, agg in stat_specs:
                key = f"{sensor}{stat_name}{axis}"
                if key not in FEATURE_COLS:
                    continue
                try:
                    val = agg(s)
                    row[key] = np.nan if (val is None or (isinstance(val, float) and np.isnan(val))) else val
                except Exception:
                    row[key] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    feature_cols_present = [c for c in FEATURE_COLS if c in out.columns]
    out = out[feature_cols_present]

    target_series = pd.Series(targets, dtype=object) if targets else None
    return out, target_series, group_keys, group_by


def get_sensor_columns(raw: pd.DataFrame, sensor_cols: dict | None = None) -> list[str]:
    """Return the 6 sensor column names in raw (AccX, AccY, AccZ, GyroX, GyroY, GyroZ or variants)."""
    if sensor_cols is None:
        sensor_cols = {}
    axes = [("Acc", "X"), ("Acc", "Y"), ("Acc", "Z"), ("Gyro", "X"), ("Gyro", "Y"), ("Gyro", "Z")]
    out = []
    for s, a in axes:
        key = (s, a)
        if key in sensor_cols and sensor_cols[key] in raw.columns:
            out.append(sensor_cols[key])
        else:
            c = _raw_col(raw, s, a)
            if c is not None:
                out.append(c)
    return out
