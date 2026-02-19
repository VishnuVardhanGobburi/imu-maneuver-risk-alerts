"""
Load CSV, dictionary xlsx, docx summary. No file uploaders; local paths only.
Tries /mnt/data then project root.
Shared get_data (cached) and ensure_data for use by app and pages without requiring home page first.
"""

from pathlib import Path
import pandas as pd

import streamlit as st

_BASE = Path(__file__).resolve().parent.parent
_ROOTS = [Path("/mnt/data"), _BASE]


def _path(name: str, ext: str) -> Path:
    p = f"{name}{ext}" if not name.endswith(ext) else name
    for root in _ROOTS:
        f = root / p
        if f.exists():
            return f
    return _BASE / p


def load_csv(path: str | None = None) -> pd.DataFrame:
    """Load main feature CSV. Keep Target + numeric columns only."""
    p = path or _path("features_14", ".csv")
    df = pd.read_csv(p)
    # Detect target column
    target_col = "Target" if "Target" in df.columns else next((c for c in df.columns if "target" in c.lower() or "label" in c.lower()), None)
    if target_col is None:
        target_col = df.columns[0]
    # Keep numeric only
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col not in numeric and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    out = df[[c for c in df.columns if c in numeric or c == target_col]].copy()
    out = out.rename(columns={target_col: "Target"}) if target_col != "Target" else out
    return out


def load_dictionary(path: str | None = None) -> pd.DataFrame | None:
    p = path or _path("Driving_AI_Full_Feature_Data_Dictionary", ".xlsx")
    if not p.exists():
        return None
    try:
        return pd.read_excel(p, sheet_name=0)
    except Exception:
        return None


def _find_sensor_raw() -> Path | None:
    """Find sensor_raw with any extension (e.g. .csv, .txt, .123, .001, .dat)."""
    for root in _ROOTS:
        base = root / "sensor_raw"
        if base.exists() and base.is_file():
            return base
        for f in root.glob("sensor_raw.*"):
            if f.is_file():
                return f
    return None


def _find_raw_sensor() -> Path | None:
    """Find raw_sensor_data with any extension (e.g. .csv, .txt, .dat)."""
    for root in _ROOTS:
        base = root / "raw_sensor_data"
        if base.exists() and base.is_file():
            return base
        for f in root.glob("raw_sensor_data.*"):
            if f.is_file():
                return f
    return None


def load_raw_sensor(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load raw_sensor_data file (e.g. raw_sensor_data.csv). Renames TaskID_New/TaskID_new to TaskID."""
    if path is not None:
        p = Path(path)
    else:
        p = _find_raw_sensor()
    if p is None or not p.exists():
        return None
    try:
        df = pd.read_csv(p, on_bad_lines="skip")
        if len(df.columns) == 1:
            df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="skip")
        # Normalize task ID column name to TaskID
        for old in ("TaskID_New", "TaskID_new"):
            if old in df.columns and "TaskID" != old:
                df = df.rename(columns={old: "TaskID"})
                break
        # Drop Unnamed columns
        cols_to_drop = [c for c in df.columns if str(c).strip().startswith("Unnamed")]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
        return df
    except Exception:
        try:
            df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="skip")
            for old in ("TaskID_New", "TaskID_new"):
                if old in df.columns and "TaskID" != old:
                    df = df.rename(columns={old: "TaskID"})
                    break
            cols_to_drop = [c for c in df.columns if str(c).strip().startswith("Unnamed")]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop, errors="ignore")
            return df
        except Exception:
            return None


def load_sensor_raw(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load raw sensor file (any extension: .csv, .123, .001, .txt, .dat). Expected columns: Task, Target (optional), AccX, AccY, AccZ, GyroX, GyroY, GyroZ."""
    if path is not None:
        p = Path(path)
    else:
        p = _find_sensor_raw()
    if p is None or not p.exists():
        return None
    try:
        df = pd.read_csv(p, on_bad_lines="skip")
        # If we got a single column, file may be tab/space-separated; try auto-detect
        if len(df.columns) == 1:
            df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="skip")
        return df
    except Exception:
        try:
            df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="skip")
            return df
        except Exception:
            return None


def load_doc_summary(path: str | None = None) -> str:
    p = path or _path("Data Collection Summary", ".docx")
    if not p.exists():
        return ""
    try:
        from docx import Document
        return "\n".join(para.text for para in Document(p).paragraphs).strip()
    except Exception:
        return ""


@st.cache_data
def get_data():
    """Load CSV, compute axis scores and class summary. Cached so any page can call without depending on home."""
    from .scores import compute_axis_scores, class_summary
    df = load_csv()
    score_df = compute_axis_scores(df)
    class_summary_df = class_summary(df, score_df)
    return df, score_df, class_summary_df


def ensure_data():
    """
    Ensure df, score_df, class_summary_df, qc are in session_state; return (df, score_df, class_summary_df, qc).
    Loads data if missing so no page depends on loading the home page first.
    """
    from .scores import quality_checks
    if st.session_state.get("df") is not None and st.session_state.get("score_df") is not None:
        return (
            st.session_state["df"],
            st.session_state["score_df"],
            st.session_state.get("class_summary_df"),
            st.session_state.get("qc"),
        )
    df, score_df, class_summary_df = get_data()
    qc = quality_checks(df, score_df)
    st.session_state["df"] = df
    st.session_state["score_df"] = score_df
    st.session_state["class_summary_df"] = class_summary_df
    st.session_state["qc"] = qc
    return df, score_df, class_summary_df, qc
