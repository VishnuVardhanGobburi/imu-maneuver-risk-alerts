"""Page 6: Maneuver prediction — Can we predict maneuver type from window features?"""

import streamlit as st
import pandas as pd
from src.io import ensure_data
from src.layout import inject_full_width
from src.predict import train_and_evaluate
from src.viz import confusion_matrix_heatmap, feature_importance_bars, TARGET_LABELS


@st.cache_data
def _cached_train(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    C: float,
    max_iter: int,
    use_ordered_split: bool,
    _cache_version: int = 5,
):
    """Cache training result. Bump _cache_version to force retrain after model changes."""
    return train_and_evaluate(
        df,
        test_size=test_size,
        random_state=random_state,
        C=C,
        max_iter=max_iter,
        use_ordered_split=use_ordered_split,
    )


def main():
    st.set_page_config(page_title="Maneuver Prediction & Alert Decision", page_icon="📊", layout="wide", initial_sidebar_state="auto")
    inject_full_width()
    df, *_ = ensure_data()
    st.markdown("## Maneuver prediction")
    st.markdown(
        "**Random Forest** is trained to predict maneuver type (1–4). "
        "Train and test splits are stratified so every class appears in both. "
        "This page shows how well the features separate maneuvers and which features matter most."
    )

    test_size, C, max_iter = 0.25, 1.0, 1000
    use_ordered_split = False
    random_state = 42

    result = _cached_train(
        df, test_size, random_state, C, max_iter, use_ordered_split, _cache_version=5
    )

    if result.get("error"):
        st.error(result["error"])
        return

    acc = result["accuracy"]
    n_train, n_test = result["n_train"], result["n_test"]
    st.metric("Test accuracy", f"{acc:.2%}", help=f"Train {n_train} rows, test {n_test}.")

    if result.get("use_ordered_split"):
        st.caption("Using ordered split: train on first rows, test on last. This often gives a more realistic accuracy.")

    tab_metrics, tab_cm, tab_importance = st.tabs(["Per-class metrics", "Confusion matrix", "Feature importance"])

    with tab_metrics:
        report = result["report_dict"]
        # Build a small table: class, precision, recall, f1, support
        rows = []
        for k, v in report.items():
            if k in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(v, dict):
                rows.append({
                    "Class": k,
                    "Precision": v.get("precision", 0),
                    "Recall": v.get("recall", 0),
                    "F1": v.get("f1-score", 0),
                    "Support": int(v.get("support", 0)),
                })
        if rows:
            metrics_df = pd.DataFrame(rows)
            # Replace class key with label if it's a digit
            metrics_df["Class"] = metrics_df["Class"].apply(
                lambda x: TARGET_LABELS.get(int(x), x) if str(x).isdigit() else x
            )
            st.dataframe(metrics_df, use_container_width=True)

    with tab_cm:
        cm = result["confusion_matrix"]
        class_names = result["class_names"]
        fig_cm = confusion_matrix_heatmap(cm, class_names, TARGET_LABELS)
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab_importance:
        importances = result["feature_importances"]
        top_n = 20
        fig_imp = feature_importance_bars(importances, top_n=top_n)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Importance = Random Forest mean decrease in impurity.")


if __name__ == "__main__":
    main()
