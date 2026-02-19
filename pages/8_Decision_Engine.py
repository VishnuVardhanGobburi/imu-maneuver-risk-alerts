import streamlit as st
import pandas as pd
from src.io import ensure_data
from src.layout import inject_full_width
from src.decision_engine import compute_di_ri, normalize_scores, run_engine
from src.viz_decision import build_gauges_or_bars, build_interaction_map, build_distribution_by_target
from src.viz import TARGET_LABELS

def main():
    st.set_page_config(page_title="Maneuver Prediction & Alert Decision", page_icon="📊", layout="wide", initial_sidebar_state="auto")
    inject_full_width()
    st.markdown("## Decision Engine (Driver vs Road Alerts)")
    st.markdown("The decision engine converts driver and road instability signals into alert decisions by identifying the primary source of risk and determining its severity.")
    df, score_df, *_ = ensure_data()
    dr = compute_di_ri(df, score_df=score_df)
    di_norm = normalize_scores(dr["DRIVER_INSTABILITY"])
    ri_norm = normalize_scores(dr["ROAD_INSTABILITY"])
    base_df = pd.DataFrame({"DI": dr["DRIVER_INSTABILITY"], "RI": dr["ROAD_INSTABILITY"], "DI_Norm": di_norm, "RI_Norm": ri_norm}, index=df.index)
    if "Target" in df.columns:
        base_df["Target"] = df["Target"].values

    if base_df.empty or base_df["DI_Norm"].isna().all():
        st.warning("No valid DI/RI or normalized scores. Ensure features_14.csv has required axis columns.")
        return

    n = len(base_df)
    target_series = df["Target"] if "Target" in df.columns else pd.Series(index=df.index, dtype=float)

    # —— Header + Big Idea ——
    st.markdown("Normalized Driver Instability(DI) and Road Instability(RI) to a common 0–1 scale so a single set of thresholds can be applied and driver versus road contributions can be compared fairly.")
    st.markdown(
        "- **DI_Norm** = how extreme driver control instability is vs typical (0–1).  \n"
        "- **RI_Norm** = how extreme road-induced vibration is vs typical (0–1).  \n"
                )
    st.divider()

    # —— Thresholds (fixed) ——
    t_low, t_med, t_dom = 0.30, 0.60, 0.20

    # —— Decision Simulator: pick an event ——
    st.markdown("#### Decision Simulator")
    row_options = list(range(n))
    sel = st.selectbox("Select a row from feature_14.csv file", row_options, format_func=lambda i: f"Row {i}", key="row_sel")
    di_val = float(base_df["DI_Norm"].iloc[sel])
    ri_val = float(base_df["RI_Norm"].iloc[sel])
    selected_idx = base_df.index[sel]

    res = run_engine(di_val, ri_val, t_low=t_low, t_med=t_med, t_dom=t_dom)

    # Live outputs (Level 0: no alert — show "No alerts Needed")
    if res is not None:
        st.markdown("**Cause:** " + res.cause + " · **Alert:** " + res.alert + " · **Severity:** " + str(res.severity))
        st.markdown("**Message:** " + res.message)
    else:
        st.markdown("**No alerts Needed**")
    st.divider()

    # —— Visuals ——
    fig_bars = build_gauges_or_bars(di_val, ri_val, title="DI_Norm & RI_Norm")
    st.plotly_chart(fig_bars, use_container_width=True)

    # Decision Engine: single red selected marker, one legend only (no maneuver colors)
    fig_map = build_interaction_map(
        ri_norm, di_norm,
        selected_idx=selected_idx,
        t_med=t_med,
        target=None,
        target_labels=None,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    rows_out = []
    for i in range(n):
        r = run_engine(base_df["DI_Norm"].iloc[i], base_df["RI_Norm"].iloc[i], t_low=t_low, t_med=t_med, t_dom=t_dom)
        if r is None:
            rows_out.append({"Cause": "", "Alert": "", "Severity": 0, "Message": ""})
        else:
            rows_out.append({"Cause": r.cause, "Alert": r.alert, "Severity": r.severity, "Message": r.message})
    out_df = base_df[["DI", "RI", "DI_Norm", "RI_Norm"]].copy()
    out_df["Cause"] = [r["Cause"] for r in rows_out]
    out_df["Alert"] = [r["Alert"] for r in rows_out]
    out_df["Severity"] = [r["Severity"] for r in rows_out]
    out_df["Message"] = [r["Message"] for r in rows_out]

    # Table/card for current selection
    st.markdown("#### Current event summary")
    row_out = out_df.iloc[sel]
    st.dataframe(row_out.to_frame().T, use_container_width=True, hide_index=True)

    # Download
    st.download_button(
        "Download decision_engine_outputs.csv",
        data=out_df.to_csv(index=True),
        file_name="decision_engine_outputs.csv",
        mime="text/csv",
        key="dl_engine",
    )

    st.divider()
    st.markdown("#### Decision engine algorithm")
    algo_html = """
    <div style="font-size: 0.9rem; margin-bottom: 1.5rem;">
      <table style="border-collapse: collapse; width: 100%; max-width: 720px;">
        <tr style="background: #f5f5f5;">
          <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Condition</th>
          <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Cause</th>
          <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Alert</th>
          <th style="text-align: center; padding: 8px; border: 1px solid #ddd;">Level</th>
        </tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">DI &amp; RI &lt; 0.30</td><td>Low (both)</td><td>No alert</td><td style="text-align: center;">0</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">(RI − DI) &gt; 0.20 and RI ≥ 0.60</td><td>Road-dominant</td><td>Road Advisory</td><td style="text-align: center;">1</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">Else: RI &gt; DI</td><td>Road-dominant</td><td>Road Advisory</td><td style="text-align: center;">1</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">Else: DI ≥ RI and DI &lt; 0.60</td><td>Driver-dominant</td><td>Driver Warning</td><td style="text-align: center;">1</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">DI &gt; 0.75 and RI &lt; 0.30</td><td>Driver-dominant</td><td>Driver Warning</td><td style="text-align: center;">2</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">RI &gt; 0.75 and DI &lt; 0.30</td><td>Road-dominant</td><td>Road Advisory</td><td style="text-align: center;">2</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">(DI − RI) &gt; 0.20 and 0.60 ≤ DI &lt; 0.80</td><td>Driver-dominant</td><td>Driver Warning</td><td style="text-align: center;">2</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">Else: DI ≥ RI and DI ≥ 0.60</td><td>Driver-dominant</td><td>Driver Warning</td><td style="text-align: center;">2</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">(DI − RI) &gt; 0.20 and DI ≥ 0.80</td><td>Driver-dominant</td><td>Driver Warning</td><td style="text-align: center;">3</td></tr>
        <tr><td style="padding: 6px; border: 1px solid #eee;">DI ≥ 0.60 and RI ≥ 0.60</td><td>Mixed</td><td>Safety Alert</td><td style="text-align: center;">3</td></tr>
      </table>
      <p style="margin-top: 8px; color: #555;">Rules are applied in order; first match wins.</p>
    </div>
    """
    st.markdown(algo_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
