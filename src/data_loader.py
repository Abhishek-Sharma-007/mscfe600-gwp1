"""
data_loader.py
--------------
Downloads, validates, and summarises daily OHLCV data from Yahoo Finance.

Usage example:
    from src.data_loader import download_data, get_data_summary
    df = download_data("ECH", "2010-01-01", "2023-12-31")
"""

import os
import yfinance as yf
import pandas as pd


# ── Core download ─────────────────────────────────────────────────────────────

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    Uses auto_adjust=True so the 'Close' column reflects split- and
    dividend-adjusted prices. Handles the MultiIndex column structure
    that newer versions of yfinance may return.

    Parameters
    ----------
    ticker : str
        ETF or stock ticker (e.g. 'ECH', 'IVV').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns Open, High, Low, Close, Volume.

    Raises
    ------
    ValueError
        If the download returns an empty DataFrame.
    """
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)

    # Newer yfinance versions return a MultiIndex — flatten it
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"between {start} and {end}. "
            "Check the ticker symbol and internet connection."
        )

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame, missing_threshold: float = 0.05) -> bool:
    """
    Run basic data quality checks.

    Parameters
    ----------
    df : pd.DataFrame
    missing_threshold : float
        Maximum allowable fraction of missing Close values.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    ValueError
        If any check fails.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    missing_frac = df["Close"].isnull().mean()
    if missing_frac > missing_threshold:
        raise ValueError(
            f"Close column has {missing_frac:.1%} missing values "
            f"(threshold: {missing_threshold:.0%})."
        )

    if (df["Close"] < 0).any():
        raise ValueError("Negative close prices detected. Check the data.")

    return True


# ── Cache helper ──────────────────────────────────────────────────────────────

def load_or_download(ticker: str, start: str, end: str,
                     cache_path: str = None) -> pd.DataFrame:
    """
    Load data from a CSV cache if it exists, otherwise download and save.

    Parameters
    ----------
    ticker, start, end : str
        As in download_data().
    cache_path : str or None
        Path to a CSV file for caching. If None, no caching is used.

    Returns
    -------
    pd.DataFrame
    """
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded cached data from: {cache_path}")
    else:
        df = download_data(ticker, start, end)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            df.to_csv(cache_path)
            print(f"Data saved to: {cache_path}")

    validate_data(df)
    return df


# ── Summary ───────────────────────────────────────────────────────────────────

def get_data_summary(df: pd.DataFrame, ticker: str) -> dict:
    """
    Build a human-readable summary dictionary for the downloaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
    ticker : str

    Returns
    -------
    dict
    """
    return {
        "Ticker":               ticker,
        "Start Date":           str(df.index[0].date()),
        "End Date":             str(df.index[-1].date()),
        "Total Trading Days":   len(df),
        "First Close (USD)":    round(float(df["Close"].iloc[0]), 4),
        "Last Close (USD)":     round(float(df["Close"].iloc[-1]), 4),
        "Min Close (USD)":      round(float(df["Close"].min()), 4),
        "Max Close (USD)":      round(float(df["Close"].max()), 4),
        "Missing Values (Close)": int(df["Close"].isnull().sum()),
    }
