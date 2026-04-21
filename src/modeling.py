"""
modeling.py
-----------
Target variable definition, feature preparation, and cross-validation pipeline.

The classifier used is logistic regression — a simple, interpretable baseline
that lets us verify the data pipeline and feature engineering before
considering more complex models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# ── Target variable ───────────────────────────────────────────────────────────

def define_target(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Append a binary classification target to the DataFrame.

    Target = +1  if the next trading day's adjusted close is higher
                  than the current day's close.
    Target = -1  otherwise.

    This formulation matches the classification label used in
    Sagaceta Mejia et al. (2022).

    Parameters
    ----------
    df : pd.DataFrame
    price_col : str

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'Target' column added.
    """
    df["Target"] = np.where(
        df[price_col].shift(-1) > df[price_col], 1, -1
    )
    return df


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Select feature and target columns, drop NaN rows, and remove the last row.

    The last row is dropped because its Target value uses shift(-1), which
    refers to a date outside the downloaded sample.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains indicator columns and 'Target'.
    features : list of str
        Names of feature columns to retain.

    Returns
    -------
    pd.DataFrame
        Clean, model-ready DataFrame.
    """
    df_m = df[features + ["Target"]].dropna()
    df_m = df_m.iloc[:-1].copy()
    return df_m


# ── Class balance ─────────────────────────────────────────────────────────────

def get_class_balance(df: pd.DataFrame) -> dict:
    """
    Summarise the class distribution of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Target' column with values +1 and -1.

    Returns
    -------
    dict
    """
    counts = df["Target"].value_counts().sort_index()
    total  = len(df)
    return {
        "Total Observations": total,
        "Up Days  (+1)":      int(counts.get(1, 0)),
        "Down Days (-1)":     int(counts.get(-1, 0)),
        "Up Days  (%)":       round(counts.get(1, 0) / total * 100, 1),
        "Down Days (%)":      round(counts.get(-1, 0) / total * 100, 1),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(X: np.ndarray, y: np.ndarray,
                          n_splits: int = 5) -> list:
    """
    Run stratified k-fold cross-validation with logistic regression.

    Features are standardised within each fold (StandardScaler fit on the
    training set only) to prevent data leakage into the test set.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary target labels (+1 / -1).
    n_splits : int
        Number of CV folds (default: 5).

    Returns
    -------
    list of dict
        Each dict contains: Fold, Train Size, Test Size, Accuracy, F1-Score.
    """
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=False)
    results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Scale within fold — fit only on training data
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)

        results.append({
            "Fold":       fold,
            "Train Size": len(tr_idx),
            "Test Size":  len(te_idx),
            "Accuracy":   round(accuracy_score(y_te, y_pred), 4),
            "F1-Score":   round(f1_score(y_te, y_pred, pos_label=1), 4),
        })

    return results
