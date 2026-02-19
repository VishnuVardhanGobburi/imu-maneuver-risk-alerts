"""
Plotly-only visuals for the Decision Engine: interaction map and DI/RI gauges or bars.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .viz import TARGET_LABELS as _TARGET_LABELS, PALETTE as _PALETTE


def _layout(fig: go.Figure, title: str = "") -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)) if title else {},
        margin=dict(t=40, b=40, l=50, r=30),
        font=dict(size=11),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )


def build_gauges_or_bars(
    di_norm: float,
    ri_norm: float,
    title: str = "DI_Norm & RI_Norm",
) -> go.Figure:
    """Side-by-side horizontal bars for DI_Norm and RI_Norm (0–1 scale)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["RI_Norm"], x=[ri_norm], orientation="h", name="RI_Norm", marker_color="rgb(52, 152, 219)", width=0.4))
    fig.add_trace(go.Bar(y=["DI_Norm"], x=[di_norm], orientation="h", name="DI_Norm", marker_color="rgb(231, 76, 60)", width=0.4))
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(range=[0, 1], title="Score (0–1)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=140,
        margin=dict(t=50, b=40, l=80, r=30),
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
    )
    return fig


def build_interaction_map(
    ri_norm: pd.Series,
    di_norm: pd.Series,
    selected_idx: int | None = None,
    t_med: float = 0.60,
    target: pd.Series | None = None,
    target_labels: dict | None = None,
) -> go.Figure:
    """
    Scatter: x=ROAD (RI_Norm), y=DRIVER (DI_Norm), 0–1 scale. Quadrant lines at t_med; four quadrant annotations.
    When target/target_labels are provided, points are colored by maneuver (4 colors). Selected point is highlighted with same maneuver color.
    """
    df = pd.DataFrame({"ROAD_INSTABILITY": ri_norm.values, "DRIVER_INSTABILITY": di_norm.values}, index=ri_norm.index)
    target_labels = target_labels or _TARGET_LABELS
    if target is not None and target_labels is not None:
        df["Target"] = target.values
        # Integer class for palette (predicted_class may be float)
        df["_class"] = df["Target"].apply(lambda x: int(x) if pd.notna(x) and x == x else None)
        df["Maneuver"] = df["_class"].map(lambda x: target_labels.get(x, str(x)) if x is not None else "—")
        df["color"] = df["_class"].map(lambda x: _PALETTE.get(x, "rgb(150,150,150)") if x is not None else "rgb(150,150,150)")
    else:
        df["Target"] = None
        df["Maneuver"] = ""
        df["color"] = "rgb(100,100,100)"

    fig = go.Figure()
    has_maneuver = target is not None and target_labels is not None

    if has_maneuver:
        # Other points: grey. Selected only: colored diamond (by maneuver). Legend: 4 diamonds for the 4 maneuvers.
        if selected_idx is not None and selected_idx in df.index:
            mask = df.index == selected_idx
            fig.add_trace(
                go.Scatter(
                    x=df.loc[~mask, "ROAD_INSTABILITY"],
                    y=df.loc[~mask, "DRIVER_INSTABILITY"],
                    mode="markers",
                    marker=dict(size=6, color="rgba(100,100,100,0.35)", line=dict(width=0)),
                    name="Other events",
                    showlegend=False,
                )
            )
            row = df.loc[selected_idx]
            sel_color = row["color"]
            fig.add_trace(
                go.Scatter(
                    x=[row["ROAD_INSTABILITY"]],
                    y=[row["DRIVER_INSTABILITY"]],
                    mode="markers",
                    marker=dict(size=14, color=sel_color, symbol="diamond", line=dict(width=2, color="white")),
                    name="Selected",
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["ROAD_INSTABILITY"],
                    y=df["DRIVER_INSTABILITY"],
                    mode="markers",
                    marker=dict(size=6, color="rgb(100,100,100)", opacity=0.6),
                    name="Events",
                    showlegend=False,
                )
            )
        # Legend: only the 4 maneuvers (diamonds with colors); points at (-1,-1) are outside axis range so not drawn
        for class_val in (1, 2, 3, 4):
            label = target_labels.get(class_val, str(class_val))
            fig.add_trace(
                go.Scatter(
                    x=[-1],
                    y=[-1],
                    mode="markers",
                    marker=dict(size=10, color=_PALETTE.get(class_val, "gray"), symbol="diamond", line=dict(width=1, color="white")),
                    name=label,
                    showlegend=True,
                    legendgroup="maneuvers",
                )
            )
    else:
        # No target: gray points, red selected
        if selected_idx is not None and selected_idx in df.index:
            mask = df.index == selected_idx
            fig.add_trace(
                go.Scatter(
                    x=df.loc[~mask, "ROAD_INSTABILITY"],
                    y=df.loc[~mask, "DRIVER_INSTABILITY"],
                    mode="markers",
                    marker=dict(size=6, color="rgba(100,100,100,0.35)", line=dict(width=0)),
                    name="Other events",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "ROAD_INSTABILITY"],
                    y=df.loc[mask, "DRIVER_INSTABILITY"],
                    mode="markers",
                    marker=dict(size=14, color="rgb(231, 76, 60)", symbol="diamond", line=dict(width=2, color="white")),
                    name="Selected event",
                    showlegend=True,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["ROAD_INSTABILITY"],
                    y=df["DRIVER_INSTABILITY"],
                    mode="markers",
                    marker=dict(size=6, color="rgb(100,100,100)", opacity=0.6),
                    name="Events",
                    showlegend=False,
                )
            )

    # Quadrant lines at t_med
    if pd.notna(t_med):
        # Vertical line at t_med (x = t_med for normalized; here we plot raw RI/DI - so we need to pass normalized or use same scale). Spec says "Quadrant lines at t_med" on the interaction map - so the map should use DI_Norm and RI_Norm for axes so t_med applies. So pass ri and di as normalized series.
        fig.add_vline(x=t_med, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
        fig.add_hline(y=t_med, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)

    fig.update_layout(
        xaxis=dict(title="ROAD (RI_Norm)", range=[-0.05, 1.05], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="DRIVER (DI_Norm)", range=[-0.05, 1.05], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        margin=dict(t=50, b=45, l=55, r=100),
        template="plotly_white",
        title=dict(text="Interaction Map (Driver vs Road)", x=0.5, xanchor="center", font=dict(size=14)),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            itemclick=False,
            itemdoubleclick=False,
        ),
        height=420,
    )
    # "Selected event" label in margin, directly above the legend box (legend at x=1.02, y=0.5)
    fig.add_annotation(
        text="Selected event",
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.68,
        showarrow=False,
        font=dict(size=12, color="#333", weight="bold"),
        xanchor="left",
        yanchor="bottom",
    )
    # Quadrant labels in corners to avoid cluttering the scatter (with light background for readability)
    pad = 0.06
    ann_opts = dict(showarrow=False, font=dict(size=9), bgcolor="rgba(255,255,255,0.88)", borderpad=4, bordercolor="rgba(0,0,0,0.12)")
    fig.add_annotation(x=pad, y=pad, text="Smooth driver / Smooth road", xanchor="left", yanchor="bottom", **ann_opts)
    fig.add_annotation(x=1 - pad, y=pad, text="Smooth driver / Rough road", xanchor="right", yanchor="bottom", **ann_opts)
    fig.add_annotation(x=pad, y=1 - pad, text="Aggressive driver / Smooth road", xanchor="left", yanchor="top", **ann_opts)
    fig.add_annotation(x=1 - pad, y=1 - pad, text="Aggressive driver / Rough road", xanchor="right", yanchor="top", **ann_opts)
    return fig


def build_distribution_by_target(
    values: pd.Series,
    target: pd.Series,
    value_name: str,
    target_labels: dict | None = None,
) -> go.Figure:
    """Single plot: distribution of value by Target (violin or box)."""
    df = pd.DataFrame({"value": values.values, "Target": target.values})
    if target_labels:
        df["Maneuver"] = df["Target"].map(lambda x: target_labels.get(x, str(x)))
        x_col = "Maneuver"
    else:
        x_col = "Target"
    fig = px.violin(df, x=x_col, y="value", box=False, points="outliers")
    fig.update_layout(
        template="plotly_white",
        title=dict(text=value_name, x=0.5, xanchor="center", font=dict(size=13)),
        xaxis_title="",
        yaxis_title=value_name,
        margin=dict(t=40, b=40, l=50, r=30),
        height=320,
    )
    return fig
