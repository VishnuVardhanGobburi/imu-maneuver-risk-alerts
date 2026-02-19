"""Download prep: axis_scores_per_row.csv, axis_scores_by_class.csv, executive_insights.md."""

import pandas as pd
from .insights import executive_summary_lines


def axis_scores_per_row_csv(score_df: pd.DataFrame, target_series: pd.Series) -> str:
    """CSV string for download."""
    out = score_df.copy()
    out.insert(0, "Target", target_series.values)
    return out.to_csv(index=True)


def axis_scores_by_class_csv(class_summary_df: pd.DataFrame) -> str:
    """CSV string for download."""
    return class_summary_df.to_csv(index=False)


def executive_insights_md(
    score_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    target_series: pd.Series,
) -> str:
    """Markdown string for executive summary download."""
    lines = executive_summary_lines(score_df, class_summary_df, target_series)
    body = "\n".join(f"- {s}" for s in lines)
    return "# Executive insights (Axis Intelligence)\n\n" + body + "\n"
