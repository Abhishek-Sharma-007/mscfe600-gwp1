"""
utils.py
--------
Shared utility functions for figure generation and general I/O.

All plot functions save high-resolution PNG files to the outputs/ directory
by default. These files are used by the Streamlit app and can be pasted
directly into the written report.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


# ── Directory helper ──────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    if path:
        os.makedirs(path, exist_ok=True)


# ── Figure 1: Price + SMA ─────────────────────────────────────────────────────

def plot_price_with_sma(df: pd.DataFrame,
                         sma_col: str = "SMA14",
                         save_path: str = "outputs/figure1_ech_sma14.png") -> None:
    """
    Plot ECH adjusted close price with 14-day SMA overlay.

    Figure is saved as a PNG file suitable for the written report and README.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' and the SMA column specified by sma_col.
    sma_col : str
    save_path : str
        Destination path (PNG). Parent directory is created if needed.
    """
    ensure_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df.index, df["Close"],
            label="ECH Adjusted Close Price",
            color="steelblue", linewidth=0.9, alpha=0.85)
    ax.plot(df.index, df[sma_col],
            label=f"{sma_col.replace('SMA','')}-Day Simple Moving Average",
            color="darkorange", linewidth=1.3)

    ax.set_title(
        "ECH iShares MSCI Chile ETF — Adjusted Close Price with 14-Day SMA (2010–2023)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price (USD)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Figure 2: RSI with thresholds ────────────────────────────────────────────

def plot_rsi(df: pd.DataFrame,
             rsi_col: str = "RSI14",
             save_path: str = "outputs/figure2_ech_rsi14.png") -> None:
    """
    Plot 14-day RSI with overbought (70) and oversold (30) threshold lines.

    Shaded regions highlight overbought/oversold zones for easier reading.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the RSI column specified by rsi_col.
    rsi_col : str
    save_path : str
    """
    ensure_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df[rsi_col],
            label="RSI (14-Day)",
            color="mediumpurple", linewidth=0.85, alpha=0.9)
    ax.axhline(70, color="crimson",  linestyle="--", linewidth=1.2,
               label="Overbought (70)")
    ax.axhline(30, color="seagreen", linestyle="--", linewidth=1.2,
               label="Oversold (30)")
    ax.fill_between(df.index, 70, 100, alpha=0.04, color="crimson")
    ax.fill_between(df.index,  0,  30, alpha=0.04, color="seagreen")

    ax.set_title(
        "ECH iShares MSCI Chile ETF — 14-Day RSI with Overbought/Oversold Thresholds (2010–2023)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("RSI Value", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Console table helper ──────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame, title: str = "") -> None:
    """Print a formatted table to the console with an optional title header."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(df.to_string(index=False))
    print()
