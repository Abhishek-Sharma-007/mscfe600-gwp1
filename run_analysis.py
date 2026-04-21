"""
run_analysis.py
---------------
Main entry-point. Runs the full ECH ETF analysis pipeline and saves
all output files (figures, CSVs, sample data) to the outputs/ directory.

Run this script before launching the Streamlit app:

    python run_analysis.py
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader   import download_data, get_data_summary
from src.indicators    import add_all_indicators, FEATURE_NAMES
from src.modeling      import define_target, prepare_features, run_cross_validation, get_class_balance
from src.evaluation    import compute_pearson_correlation, format_cv_results, save_tables
from src.text_data_demo import (create_sample_news_df, save_sample_news_csv,
                                 run_news_eda, plot_news_eda)
from src.utils         import ensure_dir, plot_price_with_sma, plot_rsi, print_summary_table


def banner(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def main() -> None:
    print("=" * 60)
    print("  MScFE 600: Financial Data — Group Work Project #1")
    print("  ECH ETF Analysis Pipeline")
    print("  Group 14434 | Oluwatobi Dahunsi, Abhishek Sharma, Sarafa Busari")
    print("=" * 60)

    ensure_dir("outputs")

    # ── 1. Download data ──────────────────────────────────────────
    banner("1 / 7  Downloading ECH ETF data")
    df = download_data("ECH", "2010-01-01", "2023-12-31")
    summary = get_data_summary(df, "ECH")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    # ── 2. Compute technical indicators ──────────────────────────
    banner("2 / 7  Computing technical indicators")
    df = add_all_indicators(df)
    print(f"  Features computed: {FEATURE_NAMES}")

    # ── 3. Target variable and feature matrix ────────────────────
    banner("3 / 7  Defining target variable")
    df       = define_target(df)
    df_model = prepare_features(df, FEATURE_NAMES)
    balance  = get_class_balance(df_model)
    for k, v in balance.items():
        print(f"  {k:<30} {v}")

    # ── 4. Pearson correlation ────────────────────────────────────
    banner("4 / 7  Pearson correlation analysis")
    corr_df = compute_pearson_correlation(df_model, FEATURE_NAMES)
    print_summary_table(corr_df, "TABLE 1 — Pearson Correlation with Target")

    # ── 5. 5-fold cross-validation ────────────────────────────────
    banner("5 / 7  5-fold cross-validation (logistic regression)")
    X  = df_model[FEATURE_NAMES].values
    y  = df_model["Target"].values
    cv = run_cross_validation(X, y, n_splits=5)
    cv_df = format_cv_results(cv)
    print_summary_table(cv_df, "TABLE 2 — Cross-Validation Results")

    # ── 6. Save tables to CSV ─────────────────────────────────────
    banner("6 / 7  Saving output tables")
    save_tables(corr_df, cv_df)

    # ── 7. Generate and save figures ──────────────────────────────
    banner("7 / 7  Generating figures")
    plot_price_with_sma(df,  save_path="outputs/figure1_ech_sma14.png")
    plot_rsi(df,             save_path="outputs/figure2_ech_rsi14.png")

    # Part 2 — news data
    news_df  = create_sample_news_df()
    save_sample_news_csv(news_df, path="data/sample_news_data.csv")
    eda      = run_news_eda(news_df)
    print(f"\n  News EDA — total records  : {eda['total_records']}")
    print(f"  News EDA — avg word count : {eda['avg_word_count']}")
    print(f"  News EDA — sentiment dist : {dict(eda['sentiment_counts'])}")
    plot_news_eda(news_df, save_path="outputs/figure3_part2_eda.png")

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All outputs saved to outputs/")
    print("  Next step: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
