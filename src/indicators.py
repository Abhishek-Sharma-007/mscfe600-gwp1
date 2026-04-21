"""
indicators.py
-------------
Functions for computing technical indicators from OHLCV price data.

Each function accepts a DataFrame and returns it with one or more new
columns appended. The top-level helper add_all_indicators() applies
all five indicators used in this project.

Indicators implemented:
    - SMA14_norm  (normalised 14-day Simple Moving Average)
    - RSI14       (14-day Relative Strength Index)
    - BB_Width    (20-day Bollinger Band Width)
    - ATR14       (14-day Average True Range)
    - ROC10       (10-day Rate of Change)
"""

import pandas as pd

# Feature names used throughout the project
FEATURE_NAMES = ["SMA14_norm", "RSI14", "BB_Width", "ATR14", "ROC10"]


# ── Individual indicator functions ────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, window: int = 14,
                price_col: str = "Close") -> pd.DataFrame:
    """
    Compute Simple Moving Average and its normalised deviation.

    SMA{window}_norm = (Close - SMA{window}) / SMA{window}

    A positive value means the current price is above its recent average
    (potential uptrend signal); negative means below (potential downtrend).

    Parameters
    ----------
    df : pd.DataFrame
    window : int
    price_col : str

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'SMA{window}' and 'SMA{window}_norm' added.
    """
    col = f"SMA{window}"
    df[col] = df[price_col].rolling(window).mean()
    df[f"{col}_norm"] = (df[price_col] - df[col]) / df[col]
    return df


def compute_rsi(df: pd.DataFrame, window: int = 14,
                price_col: str = "Close") -> pd.DataFrame:
    """
    Compute the Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss
    over a rolling window.

    Conventional thresholds: >70 overbought, <30 oversold.

    Parameters
    ----------
    df : pd.DataFrame
    window : int
    price_col : str

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'RSI{window}' added.
    """
    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)          # epsilon avoids division by zero
    df[f"RSI{window}"] = 100 - (100 / (1 + rs))
    return df


def compute_bb_width(df: pd.DataFrame, window: int = 20,
                     price_col: str = "Close") -> pd.DataFrame:
    """
    Compute Bollinger Band Width.

    BB_Width = 2 * rolling_std(window) / rolling_mean(window)

    Wider bands indicate higher volatility; narrower bands indicate
    consolidation and potential breakout conditions.

    Parameters
    ----------
    df : pd.DataFrame
    window : int
    price_col : str

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'BB_Width' added.
    """
    mid = df[price_col].rolling(window).mean()
    std = df[price_col].rolling(window).std()
    df["BB_Width"] = (2 * std) / mid
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute the Average True Range (ATR).

    True Range = max(High-Low, |High - prev.Close|, |Low - prev.Close|)
    ATR{window} = rolling mean of True Range over the window period.

    ATR measures recent price range volatility without directional bias.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain High, Low, and Close columns.
    window : int

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'ATR{window}' added.
    """
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df[f"ATR{window}"] = tr.rolling(window).mean()
    return df


def compute_roc(df: pd.DataFrame, window: int = 10,
                price_col: str = "Close") -> pd.DataFrame:
    """
    Compute the Rate of Change (ROC) as a percentage.

    ROC{window} = ((Close_t - Close_{t-n}) / Close_{t-n}) * 100

    Positive ROC signals recent upward momentum; negative signals downward.

    Parameters
    ----------
    df : pd.DataFrame
    window : int
    price_col : str

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'ROC{window}' added.
    """
    df[f"ROC{window}"] = df[price_col].pct_change(window) * 100
    return df


# ── Composite helper ──────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all five technical indicators to the DataFrame in sequence.

    After this call, the DataFrame contains:
        SMA14, SMA14_norm, RSI14, BB_Width, ATR14, ROC10

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must have Open, High, Low, Close, Volume).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with all indicator columns appended.
    """
    df = compute_sma(df, window=14)
    df = compute_rsi(df, window=14)
    df = compute_bb_width(df, window=20)
    df = compute_atr(df, window=14)
    df = compute_roc(df, window=10)
    return df
