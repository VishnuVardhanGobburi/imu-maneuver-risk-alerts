"""
All Plotly figures. Consistent maneuver colors; minimal clutter; annotations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TARGET_LABELS = {
    1: "Sudden Acceleration",
    2: "Sudden Right Turn",
    3: "Sudden Left Turn",
    4: "Sudden Brake",
}
PALETTE = {
    1: "rgb(41, 128, 185)",
    2: "rgb(231, 76, 60)",
    3: "rgb(241, 196, 15)",
    4: "rgb(46, 204, 113)",
}


def _layout(fig: go.Figure, title: str = "") -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)) if title else {},
        margin=dict(t=40, b=40, l=50, r=30),
        font=dict(size=11),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )


def maneuver_distribution(df: pd.DataFrame, target_col: str = "Target", show_title: bool = True) -> go.Figure:
    counts = df[target_col].value_counts().sort_index()
    labels = [TARGET_LABELS.get(k, str(k)) for k in counts.index]
    colors = [PALETTE.get(k, "gray") for k in counts.index]
    fig = go.Figure(go.Bar(x=labels, y=counts.values, marker_color=colors, text=counts.values))
    fig.update_traces(textposition="outside")
    _layout(fig, "Maneuver distribution" if show_title else "")
    return fig


def axis_heatmap(
    class_summary_df: pd.DataFrame,
    axes: list[str],
    score_types: list[str],
    class_val: int,
    title_suffix: str = "",
    annotate_max: bool = False,
    show_title: bool = True,
) -> go.Figure:
    """Rows=axes, Cols=score_types, values=mean for given class."""
    sub = class_summary_df[(class_summary_df["axis"].isin(axes)) & (class_summary_df["class"] == class_val)]
    if sub.empty:
        fig = go.Figure()
        _layout(fig, "No data")
        return fig
    pivot = sub.pivot(index="axis", columns="score_type", values="mean")
    for st in score_types:
        if st not in pivot.columns:
            pivot[st] = np.nan
    pivot = pivot.reindex(index=axes, columns=score_types).fillna(0)
    z_vals = pivot.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlBu_r",
            zmid=0,
            text=np.round(z_vals, 2),
            texttemplate="%{text}",
        )
    )
    if annotate_max and z_vals.size > 0 and not np.all(np.isnan(z_vals)):
        flat = np.nan_to_num(z_vals, nan=-np.inf)
        i, j = np.unravel_index(np.argmax(np.abs(flat)), z_vals.shape)
        fig.add_annotation(
            x=j, y=i, text="highest", showarrow=True, arrowhead=2, ax=30, ay=-20, font=dict(size=9),
        )
    title = f"{title_suffix} — {TARGET_LABELS.get(class_val, class_val)}" if show_title else ""
    _layout(fig, title)
    return fig


def driver_road_scatter(
    df: pd.DataFrame,
    dr: pd.DataFrame,
    target_col: str = "Target",
) -> go.Figure:
    """X=ROAD_INSTABILITY, Y=DRIVER_INSTABILITY, color=Target; quadrant lines and corner labels like Decision Engine."""
    plot_df = df[[target_col]].copy()
    plot_df["ROAD_INSTABILITY"] = dr["ROAD_INSTABILITY"].values
    plot_df["DRIVER_INSTABILITY"] = dr["DRIVER_INSTABILITY"].values
    plot_df["Maneuver"] = plot_df[target_col].map(TARGET_LABELS)
    fig = px.scatter(
        plot_df,
        x="ROAD_INSTABILITY",
        y="DRIVER_INSTABILITY",
        color="Maneuver",
        color_discrete_map={TARGET_LABELS[k]: PALETTE[k] for k in (1, 2, 3, 4)},
        opacity=0.7,
    )
    x_min, x_max = plot_df["ROAD_INSTABILITY"].min(), plot_df["ROAD_INSTABILITY"].max()
    y_min, y_max = plot_df["DRIVER_INSTABILITY"].min(), plot_df["DRIVER_INSTABILITY"].max()
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    pad_x, pad_y = 0.06 * x_range, 0.06 * y_range
    ann_opts = dict(showarrow=False, font=dict(size=9), bgcolor="rgba(255,255,255,0.88)", borderpad=4, bordercolor="rgba(0,0,0,0.12)")
    fig.add_annotation(x=x_min + pad_x, y=y_min + pad_y, text="Smooth driver / Smooth road", xanchor="left", yanchor="bottom", **ann_opts)
    fig.add_annotation(x=x_max - pad_x, y=y_min + pad_y, text="Smooth driver / Rough road", xanchor="right", yanchor="bottom", **ann_opts)
    fig.add_annotation(x=x_min + pad_x, y=y_max - pad_y, text="Aggressive driver / Smooth road", xanchor="left", yanchor="top", **ann_opts)
    fig.add_annotation(x=x_max - pad_x, y=y_max - pad_y, text="Aggressive driver / Rough road", xanchor="right", yanchor="top", **ann_opts)
    _layout(fig, "Driver vs Road attribution")
    return fig


def violin_driver_instability(df: pd.DataFrame, dr: pd.DataFrame, target_col: str = "Target") -> go.Figure:
    plot_df = df[[target_col]].copy()
    plot_df["DRIVER_INSTABILITY"] = dr["DRIVER_INSTABILITY"].values
    plot_df["Maneuver"] = plot_df[target_col].map(TARGET_LABELS)
    order = [TARGET_LABELS[i] for i in sorted(plot_df[target_col].unique())]
    fig = px.violin(
        plot_df,
        x="Maneuver",
        y="DRIVER_INSTABILITY",
        color="Maneuver",
        color_discrete_map={TARGET_LABELS[k]: PALETTE[k] for k in (1, 2, 3, 4)},
        category_orders={"Maneuver": order},
    )
    fig.update_layout(showlegend=False, yaxis_title="Driver Instability")
    _layout(fig, "Driver Instability by maneuver")
    return fig


def violin_road_instability(df: pd.DataFrame, dr: pd.DataFrame, target_col: str = "Target") -> go.Figure:
    plot_df = df[[target_col]].copy()
    plot_df["ROAD_INSTABILITY"] = dr["ROAD_INSTABILITY"].values
    plot_df["Maneuver"] = plot_df[target_col].map(TARGET_LABELS)
    order = [TARGET_LABELS[i] for i in sorted(plot_df[target_col].unique())]
    fig = px.violin(
        plot_df,
        x="Maneuver",
        y="ROAD_INSTABILITY",
        color="Maneuver",
        color_discrete_map={TARGET_LABELS[k]: PALETTE[k] for k in (1, 2, 3, 4)},
        category_orders={"Maneuver": order},
    )
    fig.update_layout(showlegend=False, yaxis_title="Road Instability")
    _layout(fig, "Road Instability by maneuver")
    return fig


def bias_violin_turning_only(
    df: pd.DataFrame,
    score_df: pd.DataFrame,
    target_col: str = "Target",
    score_col: str = "GyroZ_Directional_Bias",
) -> go.Figure:
    """Bias distribution for turning classes (2, 3) only."""
    turn_mask = df[target_col].isin([2, 3])
    plot_df = df.loc[turn_mask, [target_col]].copy()
    plot_df["bias"] = score_df.loc[turn_mask, score_col].values if score_col in score_df.columns else np.nan
    plot_df["Maneuver"] = plot_df[target_col].map(TARGET_LABELS)
    plot_df = plot_df.dropna(subset=["bias"])
    if plot_df.empty:
        return go.Figure()
    fig = px.violin(
        plot_df,
        x="Maneuver",
        y="bias",
        color="Maneuver",
        color_discrete_map={TARGET_LABELS[2]: PALETTE[2], TARGET_LABELS[3]: PALETTE[3]},
    )
    display_label = "Gyro Directional Bias" if "Gyro" in score_col else "Directional Bias"
    fig.update_layout(showlegend=False, yaxis_title=display_label)
    _layout(fig, f"{display_label} (turns only)")
    return fig


def scorecard_grouped_bars(row: pd.Series, axes: list[str], show_title: bool = True) -> go.Figure:
    """Grouped bar: Intensity, Variability, Impulsiveness, Directional_Bias per axis (e.g. Insight Generator)."""
    metrics = [
        ("Intensity", "#2980b9"),
        ("Variability", "#27ae60"),
        ("Impulsiveness", "#e67e22"),
        ("Directional_Bias", "#8e44ad"),
    ]
    x_axis = []
    data = {m[0]: [] for m in metrics}
    for ax in axes:
        if f"{ax}_Intensity" not in row:
            continue
        x_axis.append(ax)
        for name, _ in metrics:
            key = f"{ax}_{name}" if name != "Directional_Bias" else f"{ax}_Directional_Bias"
            val = row.get(key, 0)
            data[name].append((float(val) if pd.notna(val) else 0) or 0)
    if not x_axis:
        return go.Figure()
    fig = go.Figure()
    for name, color in metrics:
        fig.add_trace(go.Bar(name=name.replace("_", " "), x=x_axis, y=data[name], marker_color=color))
    fig.update_layout(barmode="group")
    _layout(fig, "Axis Intensity, Variability, Impulsiveness & Directional Bias" if show_title else "")
    return fig


def axis_distribution_histograms(
    score_df: pd.DataFrame,
    mask: pd.Series,
    axes: list[str],
    score_cols: list[str],
) -> go.Figure:
    """6×4 grid of histograms: rows = axes (AccX..GyroZ), cols = score types (Intensity, Variability, Impulsiveness, Directional_Bias)."""
    nrows, ncols = len(axes), len(score_cols)
    titles = [f"{ax} — {sc}" for ax in axes for sc in score_cols]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=titles,
        vertical_spacing=0.075,
        horizontal_spacing=0.06,
    )
    for i, ax in enumerate(axes):
        for j, sc in enumerate(score_cols):
            col = f"{ax}_{sc}"
            if col not in score_df.columns:
                continue
            data = score_df.loc[mask, col].dropna()
            if len(data) == 0:
                continue
            fig.add_trace(
                go.Histogram(x=data, nbinsx=min(30, max(10, len(data) // 5)), showlegend=False),
                row=i + 1,
                col=j + 1,
            )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=78, b=45, l=50, r=30),
        height=900,
    )
    fig.update_annotations(font_size=10)
    fig.update_xaxes(matches=None, showticklabels=True, tickfont_size=9)
    fig.update_yaxes(matches=None, showticklabels=True, tickfont_size=9)
    return fig


def confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list[int],
    target_labels: dict[int, str] | None = None,
) -> go.Figure:
    """Heatmap of confusion matrix with class labels."""
    labels = target_labels or {}
    y_labels = [labels.get(c, str(c)) for c in class_names]
    x_labels = y_labels
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=x_labels,
            y=y_labels,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Confusion matrix (test set)", x=0.5, xanchor="center"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(size=11),
    )
    return fig


def feature_importance_bars(importances: pd.Series, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top N feature importances."""
    top = importances.head(top_n)
    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker_color="rgb(52, 152, 219)",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"Top {top_n} feature importances", x=0.5, xanchor="center"),
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(t=40, b=40, l=120, r=40),
        height=max(400, top_n * 18),
        yaxis=dict(autorange="reversed"),
    )
    return fig
