"""
evaluation.py
-------------
Pearson correlation analysis, cross-validation result formatting,
and CSV table export utilities.
"""

import os
import pandas as pd


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_pearson_correlation(df: pd.DataFrame, features: list,
                                 target_col: str = "Target") -> pd.DataFrame:
    """
    Compute the Pearson correlation between each feature and the target.

    This is the descriptive statistics step — it summarises linear associations
    without making predictions. It is used here instead of LASSO as the simpler
    alternative metric specified in the replication step.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
    target_col : str

    Returns
    -------
    pd.DataFrame
        Columns: Feature, Pearson Corr. with Target.
    """
    corr_vals = df[features].corrwith(df[target_col])
    return pd.DataFrame({
        "Feature":                    features,
        "Pearson Corr. with Target":  corr_vals.values.round(4),
    })


# ── CV result formatting ──────────────────────────────────────────────────────

def format_cv_results(cv_results: list) -> pd.DataFrame:
    """
    Convert a list of fold-result dicts into a clean DataFrame with a mean row.

    Parameters
    ----------
    cv_results : list of dict
        Output from modeling.run_cross_validation().

    Returns
    -------
    pd.DataFrame
        One row per fold plus a final mean row.
    """
    df      = pd.DataFrame(cv_results)
    mean_row = pd.DataFrame([{
        "Fold":       "Mean",
        "Train Size": "—",
        "Test Size":  "—",
        "Accuracy":   round(df["Accuracy"].mean(), 4),
        "F1-Score":   round(df["F1-Score"].mean(), 4),
    }])
    return pd.concat([df, mean_row], ignore_index=True)


# ── CSV export ────────────────────────────────────────────────────────────────

def save_tables(corr_df: pd.DataFrame, cv_df: pd.DataFrame,
                output_dir: str = "outputs") -> None:
    """
    Save correlation and CV-results tables as CSV files.

    Files created:
        outputs/correlation_table.csv
        outputs/cv_results_table.csv

    Parameters
    ----------
    corr_df : pd.DataFrame
    cv_df   : pd.DataFrame
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    corr_path = os.path.join(output_dir, "correlation_table.csv")
    cv_path   = os.path.join(output_dir, "cv_results_table.csv")

    corr_df.to_csv(corr_path, index=False)
    cv_df.to_csv(cv_path,   index=False)

    print(f"Saved: {corr_path}")
    print(f"Saved: {cv_path}")
