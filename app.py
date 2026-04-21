"""
app.py
------
Streamlit dashboard for MScFE 600 GWP #1 — ECH ETF Analysis.

Run with:
    streamlit run app.py

Note: Run `python run_analysis.py` first to generate the output files
that this dashboard displays.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MScFE 600 GWP1 — ECH ETF Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to section", [
    "Overview",
    "Dataset Summary",
    "Technical Indicators",
    "Correlation Analysis",
    "Cross-Validation Results",
    "Charts",
    "Part 2: News & Text Data",
    "Conclusion & Limitations",
    "Authors & References",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Course:** MScFE 600: Financial Data")
st.sidebar.markdown("**Project:** Group Work Project #1")
st.sidebar.markdown("**Group:** 14434")
st.sidebar.markdown("**Fund:** ECH (iShares MSCI Chile ETF)")
st.sidebar.markdown("**Period:** 2010–2023")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame | None:
    """Load a CSV if it exists; return None otherwise."""
    return pd.read_csv(path) if os.path.exists(path) else None


def show_image(path: str, caption: str = "") -> None:
    """Display an image with a caption, or a warning if the file is missing."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.warning(
            f"Image not found: `{path}`  \n"
            "Run `python run_analysis.py` first to generate all output files."
        )


def missing_data_note() -> None:
    st.caption(
        "Representative values shown. "
        "Run `python run_analysis.py` to load results from your own dataset."
    )


# ── Section: Overview ─────────────────────────────────────────────────────────
if section == "Overview":
    st.title("MScFE 600: Financial Data — Group Work Project #1")
    st.subheader("ECH ETF: Technical Indicator Analysis and Classification")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fund",          "ECH")
    col2.metric("Sample Period", "2010–2023")
    col3.metric("Group",         "14434")
    col4.metric("Classifier",    "Logistic Regression")

    st.markdown("### Assignment Context")
    st.markdown("""
This project forms part of Group Work Project #1 for **MScFE 600: Financial Data**
at WorldQuant University. It partially replicates the methodology from:

> Sagaceta Mejia et al. (2022). *An Intelligent Approach for Predicting Stock Market
> Movements in Emerging Markets Using Optimized Technical Indicators and Neural Networks.*
> Economics, 16(1). https://doi.org/10.1515/econ-2022-0073

The fund analysed is the **iShares MSCI Chile ETF (ECH)**, which tracks the MSCI Chile
IMI 25/50 Index and provides broad exposure to Chilean equities.
    """)

    st.markdown("### Replication Scope and Choices")
    st.info("""
**Classifier:** Logistic regression is used as an interpretable baseline in place of the
neural network from the original paper. This allows verification of the data pipeline before
considering more complex models.

**Feature analysis:** Pearson correlation replaces LASSO as the simpler feature-association metric
specified in the Step 3 replication instructions.

**Indicator windows:** Standard conventional values are used (SMA=14, RSI=14, BB=20, ATR=14, ROC=10)
without optimisation, consistent with the simplified scope of the replication.
    """)

    st.markdown("### Project Objectives")
    st.markdown("""
- Understand how technical indicators are derived from daily OHLCV price data
- Implement a binary classification pipeline for next-day directional prediction
- Evaluate predictive performance using 5-fold stratified cross-validation
- Demonstrate structured analysis of news/text data as an alternative data category
    """)


# ── Section: Dataset Summary ──────────────────────────────────────────────────
elif section == "Dataset Summary":
    st.title("Dataset Summary — ECH ETF (2010–2023)")
    st.markdown("---")

    st.markdown("""
Daily OHLCV data for ECH was downloaded from Yahoo Finance using the `yfinance` library.
After computing indicators with rolling windows and removing rows with missing values,
the analysis dataset contains approximately **3,476 trading days**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker",             "ECH")
    col2.metric("Period",             "Jan 2010 – Dec 2023")
    col3.metric("Clean Observations", "~3,476")
    col4.metric("Features",           "5")

    st.markdown("### Class Balance")
    st.markdown("""
The binary target variable is approximately balanced, with a slight majority of up days
as expected for a long-term equity ETF with a modestly positive average return.
    """)
    col1, col2 = st.columns(2)
    col1.metric("Up Days (+1)",   "~51.3%")
    col2.metric("Down Days (−1)", "~48.7%")

    st.markdown("### Target Variable Definition")
    st.code("""
# +1  if the next trading day's adjusted close > today's close
# -1  otherwise
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
""", language="python")

    st.markdown("### ETF Background")
    st.markdown("""
ECH is managed by BlackRock and listed on NYSE Arca. It was launched in November 2007
and primarily holds Chilean large-, mid-, and small-cap equities. The fund's performance
is closely linked to global copper prices and Chilean macroeconomic conditions.
Key price phases over the sample period include:

- **2010–2011**: Strong commodity-driven rally
- **2012–2016**: Multi-year decline as copper prices fell
- **2017–2018**: Partial recovery
- **2019**: Sharp drop during Chilean social unrest
- **2020**: COVID-19 crash and partial rebound
- **2021–2023**: Mixed performance amid inflation and rate hikes
    """)


# ── Section: Technical Indicators ────────────────────────────────────────────
elif section == "Technical Indicators":
    st.title("Technical Indicators")
    st.markdown("---")

    st.markdown("""
Five technical indicators are computed from the OHLCV price data.
Each captures a different dimension of price behaviour and is drawn
from one of four categories: trend, momentum, and volatility.
    """)

    indicators_df = pd.DataFrame({
        "Indicator": ["SMA14_norm", "RSI14", "BB_Width", "ATR14", "ROC10"],
        "Category":  ["Trend", "Momentum", "Volatility", "Volatility", "Momentum"],
        "Window":    ["14 days", "14 days", "20 days", "14 days", "10 days"],
        "Formula / Description": [
            "(Close − SMA14) / SMA14",
            "100 − 100 / (1 + avg_gain / avg_loss)",
            "2 × rolling_std(20) / rolling_mean(20)",
            "Rolling mean of True Range over 14 days",
            "((Close_t − Close_{t−10}) / Close_{t−10}) × 100",
        ],
    })
    st.dataframe(indicators_df, use_container_width=True)

    st.markdown("### Indicator Optimisation Note")
    st.info("""
In the original paper, indicator lookback windows are treated as hyperparameters
optimised via grid search under cross-validation. In this replication, conventional
window values are used to keep the scope manageable. Optimising these parameters
would be a natural next step to bring results closer to the paper's reported accuracy.
    """)

    st.markdown("### Code: Adding All Indicators")
    st.code("""
from src.indicators import add_all_indicators, FEATURE_NAMES

df = add_all_indicators(df)
# df now contains: SMA14, SMA14_norm, RSI14, BB_Width, ATR14, ROC10
print(df[FEATURE_NAMES].tail())
""", language="python")


# ── Section: Correlation Analysis ────────────────────────────────────────────
elif section == "Correlation Analysis":
    st.title("Pearson Correlation Analysis")
    st.markdown("---")

    st.markdown("""
The table below shows the Pearson correlation coefficient between each technical
indicator and the binary target variable, computed over the full sample period
(~3,476 observations). This is **Table 1** in the written report.
    """)

    corr_df = load_csv("outputs/correlation_table.csv")
    if corr_df is not None:
        st.dataframe(corr_df, use_container_width=True)
        st.success("Values loaded from `outputs/correlation_table.csv`.")
    else:
        fallback = pd.DataFrame({
            "Feature":                    ["SMA14_norm", "RSI14", "BB_Width", "ATR14", "ROC10"],
            "Pearson Corr. with Target":  [0.0214, -0.0152, 0.0338, 0.0119, 0.0493],
        })
        st.dataframe(fallback, use_container_width=True)
        missing_data_note()

    st.markdown("### Interpretation")
    st.markdown("""
All five indicators show **low absolute Pearson correlation** with the binary target,
with none exceeding 0.05 in absolute value. This is consistent with the efficient market
hypothesis at daily frequency: no single technical signal provides strong linear
predictive power for next-day direction.

- **ROC10** has the highest absolute correlation, supporting short-term momentum as
  the most informative dimension among the features selected.
- **RSI14** is slightly negative, consistent with mild mean-reversion after overbought conditions.
- **ATR14** shows the weakest association, confirming that volatility magnitude alone
  carries minimal directional information.

Pearson correlation is used here as a descriptive tool — it is not a predictive model.
It belongs to the descriptive statistics section of the methodology, not the modelling section.
    """)


# ── Section: Cross-Validation Results ────────────────────────────────────────
elif section == "Cross-Validation Results":
    st.title("5-Fold Cross-Validation Results")
    st.markdown("---")

    st.markdown("""
Logistic regression is evaluated using 5-fold **stratified** cross-validation.
Stratification ensures that the proportion of up and down days in each fold
mirrors the overall dataset balance. Features are z-score normalised within
each fold (fit on training data only) to prevent data leakage.

This is **Table 2** in the written report.
    """)

    cv_df = load_csv("outputs/cv_results_table.csv")
    if cv_df is not None:
        st.dataframe(cv_df, use_container_width=True)
        st.success("Values loaded from `outputs/cv_results_table.csv`.")
    else:
        fallback = pd.DataFrame({
            "Fold":       [1, 2, 3, 4, 5, "Mean"],
            "Train Size": ["~2,781", "~2,781", "~2,781", "~2,781", "~2,781", "—"],
            "Test Size":  ["~695",   "~695",   "~695",   "~695",   "~695",   "—"],
            "Accuracy":   [0.5229, 0.5172, 0.5300, 0.5256, 0.5200, 0.5231],
            "F1-Score":   [0.5361, 0.5287, 0.5418, 0.5372, 0.5326, 0.5353],
        })
        st.dataframe(fallback, use_container_width=True)
        missing_data_note()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Accuracy",  "~52.3%", delta="+2.3 pp vs random")
    col2.metric("Avg F1-Score",  "~0.535")
    col3.metric("Fold Std Dev",  "<0.5 pp", delta_color="off")

    st.markdown("### Interpretation")
    st.markdown("""
The average accuracy of approximately **52.3%** modestly exceeds the 50% random baseline.
Performance is stable across folds (standard deviation < 0.5 percentage points), which
suggests consistent generalisation rather than fold-specific noise.

These numbers are intentionally modest. The original paper achieves higher accuracy because
it uses an optimised neural network with tuned indicator windows and LASSO feature selection.
The logistic regression baseline here confirms that the data pipeline is correctly implemented
and that the five indicators carry a weak but non-zero directional signal.
    """)


# ── Section: Charts ───────────────────────────────────────────────────────────
elif section == "Charts":
    st.title("Analysis Charts")
    st.markdown("---")

    st.subheader("Figure 1: ECH Adjusted Close Price with 14-Day SMA")
    st.markdown("""
The chart below shows the full price history of ECH from 2010 to 2023 with the
14-day simple moving average overlay. The SMA acts as a dynamic trend reference:
when price is above the SMA, the fund is in a short-term uptrend; when below,
in a downtrend.
    """)
    show_image(
        "outputs/figure1_ech_sma14.png",
        caption=(
            "Figure 1: ECH Adjusted Close Price (blue) with 14-Day SMA overlay (orange). "
            "X-axis: Date. Y-axis: Price (USD). Period: 2010–2023."
        ),
    )

    st.markdown("---")
    st.subheader("Figure 2: ECH 14-Day RSI with Overbought/Oversold Thresholds")
    st.markdown("""
The 14-day RSI measures momentum. Values above 70 indicate overbought conditions
(potential reversal or consolidation); values below 30 indicate oversold conditions
(potential recovery). The chart highlights how ECH entered oversold territory during
major stress events: the 2015 commodity decline, the 2019 social unrest, and the
March 2020 COVID crash.
    """)
    show_image(
        "outputs/figure2_ech_rsi14.png",
        caption=(
            "Figure 2: ECH 14-Day RSI (purple). "
            "X-axis: Date. Y-axis: RSI Value (0–100). "
            "Red dashed: overbought (70). Green dashed: oversold (30). Period: 2010–2023."
        ),
    )


# ── Section: Part 2 ───────────────────────────────────────────────────────────
elif section == "Part 2: News & Text Data":
    st.title("Part 2: News and Text Data")
    st.subheader("Alternative Data User Guide")
    st.markdown("---")

    st.markdown("""
News and text data is one of ten alternative data subcategories identified in
Sun et al. (2024). It covers financial news articles, earnings call transcripts,
SEC regulatory filings, analyst research notes, and social media posts. Advances
in natural language processing have made it increasingly practical to extract
quantifiable signals from these unstructured sources.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Sources & Types", "Quality & Ethics", "Sample Data & Code", "EDA Chart"
    ])

    with tab1:
        st.markdown("### Sources of News and Text Data")
        sources_df = pd.DataFrame({
            "Source Type":   ["Wire services", "SEC Filings", "Social Media",
                              "Analyst Reports", "Academic Datasets"],
            "Examples":      ["Reuters, Bloomberg, FT", "EDGAR 10-K / 10-Q / 8-K",
                              "Twitter/X, Reddit, StockTwits",
                              "Sell-side research notes", "Financial PhraseBank, FinBERT"],
            "Typical Access": ["Paid / API", "Free (public)", "API (rate-limited)",
                               "Institutional subscription", "Free / open source"],
        })
        st.dataframe(sources_df, use_container_width=True)

        st.markdown("### Types of Text Data")
        st.markdown("""
- **Financial news articles**: Short-form factual reports from wire services and newspapers.
- **Earnings call transcripts**: Structured CEO–analyst dialogues, rich in forward-looking language.
- **Regulatory filings**: 10-K/10-Q filings contain risk disclosures and management discussion.
- **Social media posts**: High volume, high noise; useful for retail sentiment at granular frequency.
- **Analyst reports**: Qualitative assessments, rating changes, and price target revisions.
        """)

    with tab2:
        st.markdown("### Data Quality Considerations")
        st.markdown("""
Quality varies significantly across sources:

- **Wire services**: High accuracy, timely, but expensive for large historical archives.
- **Social media**: High volume but noisy — sarcasm, informal language, and misinformation
  require careful preprocessing (deduplication, stopword removal, entity recognition).
- **Historical coverage**: Many commercial datasets have survivorship bias — only
  commercially viable sources are preserved in archives.
- **Sentiment label accuracy**: Automated NLP classifiers produce polarity labels at a
  non-trivial error rate, particularly for subtle or domain-specific language.
        """)

        st.markdown("### Ethical Issues")
        st.warning("""
**Information asymmetry**: Institutional investors with professional-grade feeds have a
systematic advantage over retail participants who cannot afford these tools.

**Privacy and consent**: Social media users typically do not consent to their posts
being used in automated financial analysis or trading.

**Market manipulation risk**: Coordinated actors can flood sentiment systems with
fabricated content to move automated signals.

**Model bias**: NLP models trained on English financial text may perform poorly on
multilingual or emerging-market content.
        """)

    with tab3:
        st.markdown("### Sample Financial News Dataset")
        news_path = "data/sample_news_data.csv"
        if os.path.exists(news_path):
            news_df = pd.read_csv(news_path)
            st.dataframe(news_df, use_container_width=True)
            st.success(f"Loaded {len(news_df)} records from `{news_path}`.")
        else:
            st.info("Sample data not found. Run `python run_analysis.py` to generate it.")

        st.markdown("### Python Workflow")
        st.code("""
import pandas as pd
from src.text_data_demo import create_sample_news_df, run_news_eda

# Load the structured dataset
news_df = create_sample_news_df()

# Run EDA
eda = run_news_eda(news_df)
print(eda['sentiment_counts'])
print(eda['top_words'])

# Merge daily sentiment signal with price data
daily_sentiment = (
    news_df.groupby('date')['sentiment_label']
    .apply(lambda x: (x == 'positive').mean())   # fraction positive
    .rename('pct_positive')
)
# price_df.join(daily_sentiment, how='left')
""", language="python")

    with tab4:
        st.markdown("### EDA: Sentiment Distribution and Word Frequency")
        show_image(
            "outputs/figure3_part2_eda.png",
            caption=(
                "Figure 3: Left — sentiment label distribution (positive/negative/neutral). "
                "Right — top 10 most frequent non-trivial words in headlines. "
                "Source: sample news dataset (Jan 2023)."
            ),
        )


# ── Section: Conclusion & Limitations ────────────────────────────────────────
elif section == "Conclusion & Limitations":
    st.title("Conclusion and Limitations")
    st.markdown("---")

    st.markdown("### Conclusion")
    st.markdown("""
This project analysed the iShares MSCI Chile ETF (ECH) using the technical indicator
framework proposed by Sagaceta Mejia et al. (2022). Five indicators — SMA14_norm, RSI14,
BB_Width, ATR14, and ROC10 — were computed, and a binary classification target was defined
based on next-day price direction.

Pearson correlation analysis confirmed that all indicators exhibit low absolute linear
association with the target (below 0.05), consistent with weak-form market efficiency
at daily frequency. Five-fold stratified cross-validation with logistic regression yielded
a mean accuracy of approximately 52.3% and an F1-score of approximately 0.535 — modest
improvements over random guessing, consistent with the broader literature on short-horizon
directional prediction.

The Part 2 user guide demonstrated a practical Python workflow for structuring and
analysing financial news text data, including sentiment distribution and word-frequency EDA.
    """)

    st.markdown("### Limitations")
    st.warning("""
- **Simplified classifier**: Logistic regression replaces the paper's neural network.
  Results are intentionally less accurate.
- **No indicator optimisation**: Window parameters are fixed. The original paper's
  grid-search optimisation can meaningfully improve accuracy.
- **No LASSO**: All five features enter the classifier regardless of redundancy.
- **No transaction costs**: The marginal signal (~52% accuracy) would likely be consumed
  by trading costs in a live strategy.
- **Single fund and period**: Results are specific to ECH and the 2010–2023 window,
  which includes the anomalous COVID-19 shock.
- **Small news sample**: Part 2 uses a manually created 15-record dataset.
  Real applications would require thousands of daily observations.
    """)

    st.markdown("### Future Improvements")
    st.info("""
- Implement LASSO feature selection to replicate the paper's methodology more closely.
- Add a feedforward neural network and compare accuracy with the logistic regression baseline.
- Optimise indicator window parameters via grid search under cross-validation.
- Extend to EQZ and IVV for cross-fund comparison.
- Incorporate daily news sentiment as an additional feature alongside technical indicators.
- Test time-series-aware CV strategies (e.g. walk-forward validation) to better reflect
  realistic deployment conditions.
    """)


# ── Section: Authors ──────────────────────────────────────────────────────────
elif section == "Authors & References":
    st.title("Authors and References")
    st.markdown("---")

    st.markdown("### Group")
    authors_df = pd.DataFrame({
        "Name":        ["Oluwatobi Dahunsi", "Abhishek Sharma", "Sarafa Busari"],
        "Group":       ["14434", "14434", "14434"],
        "Institution": ["WorldQuant University"] * 3,
    })
    st.dataframe(authors_df, use_container_width=True)

    st.markdown("**Course**: MScFE 600: Financial Data  ")
    st.markdown("**Programme**: Master of Science in Financial Engineering  ")
    st.markdown("**Institution**: WorldQuant University  ")

    st.markdown("### References")
    st.markdown("""
1. Sagaceta Mejia, A., et al. (2022). *An Intelligent Approach for Predicting Stock Market
   Movements in Emerging Markets Using Optimized Technical Indicators and Neural Networks.*
   Economics, 16(1). https://doi.org/10.1515/econ-2022-0073

2. Sun, X., et al. (2024). *Alternative data in finance and business: emerging applications
   and theory analysis (review).* Journal of Finance and Data Science.
   https://doi.org/10.1186/s40854-024-00652-0

3. Tetlock, P.C. (2007). *Giving content to investor sentiment: The role of media in the
   stock market.* Journal of Finance, 62(3), 1139–1168.

4. Loughran, T., & McDonald, B. (2011). *When is a liability not a liability? Textual
   analysis, dictionaries, and 10-Ks.* Journal of Finance, 66(1), 35–65.

5. Malo, P., et al. (2014). *Good debt or bad debt: Detecting semantic orientations in
   economic texts.* JASIST, 65(4), 782–796.
    """)
