"""
Comprehensive Visualization and Chart Generation Suite.
Generates publication-quality charts, statistical plots, and machine learning
evaluation curves for the Online Shopping Purchase Intention Prediction project.
Saves all figures in high-resolution (300 DPI) to reports/visualizations/.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)

from config import (
    DATA_FILE_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    VISUALIZATIONS_DIR,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
)
from src.data_loader import load_raw_data
from src.preprocessing import prepare_data, transform_features
from src.utils import logger, load_artifact

# Configure aesthetic visual styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "bold"

# Professional Theme Color Palette
NAVY = "#1F4E79"
BLUE = "#2563EB"
GREEN = "#16A34A"
RED = "#DC2626"
AMBER = "#F59E0B"
PURPLE = "#7C3AED"
SLATE = "#64748B"
LIGHT_BG = "#F8FAFC"


def setup_canvas(nrows: int = 1, ncols: int = 1, figsize: tuple = (8, 5)):
    """Helper to create figure and axis with consistent styling."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    return fig, axes


def save_chart(fig, filename: str, extra_aliases: List[str] = None):
    """Save figure with high resolution and create required aliases."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    primary_path = VISUALIZATIONS_DIR / filename
    fig.tight_layout()
    fig.savefig(primary_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    if extra_aliases:
        for alias in extra_aliases:
            alias_path = VISUALIZATIONS_DIR / alias
            import shutil
            shutil.copyfile(primary_path, alias_path)

    logger.info(f"Generated chart: {primary_path.name}")
    return primary_path


# ==============================================================================
# 1. TARGET CLASS DISTRIBUTION
# ==============================================================================
def generate_target_distribution(df: pd.DataFrame):
    fig, ax = setup_canvas(figsize=(7, 4.8))
    counts = df[TARGET_COLUMN].value_counts()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100

    labels = ["No Purchase", "Completed Purchase"]
    vals = [counts.get("No", 0), counts.get("Yes", 0)]
    colors = ["#94A3B8", GREEN]

    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="#0F172A", linewidth=1.2)

    for bar, val in zip(bars, vals):
        pct = (val / len(df)) * 100
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,} sessions\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Target Class Distribution: Completed Purchase vs. No Purchase", pad=15)
    ax.set_ylabel("Number of Survey Respondents")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Imbalance ratio annotation
    ratio = counts.get("No", 1) / max(1, counts.get("Yes", 1))
    ax.text(
        0.98,
        0.92,
        f"Sample: N = {len(df):,}\nClass Imbalance: {ratio:.2f} : 1\nConversion Rate: {percentages.get('Yes', 0):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_BG, edgecolor=SLATE, alpha=0.9),
        fontsize=9.5,
    )

    return save_chart(
        fig,
        "01_target_distribution.png",
        ["target_distribution.png"],
    )


# ==============================================================================
# 2. DEMOGRAPHICS (AGE & GENDER)
# ==============================================================================
def generate_demographics_analysis(df: pd.DataFrame):
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(13, 5))

    # Subplot 1: Age Histogram + KDE by Purchase
    sns.histplot(
        data=df,
        x="Age",
        hue=TARGET_COLUMN,
        palette={"No": "#94A3B8", "Yes": GREEN},
        multiple="stack",
        bins=20,
        kde=True,
        ax=ax1,
        edgecolor="#0F172A",
    )
    ax1.set_title("Age Distribution Stratified by Purchase Outcome")
    ax1.set_xlabel("Shopper Age (Years)")
    ax1.set_ylabel("Respondent Count")

    # Subplot 2: Gender Conversion Rate
    gender_stats = (
        df.groupby("Gender")[TARGET_COLUMN]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="Conversion_Rate")
    )
    bars = ax2.bar(
        gender_stats["Gender"],
        gender_stats["Conversion_Rate"],
        color=[NAVY, BLUE],
        width=0.45,
        edgecolor="#0F172A",
        linewidth=1.0,
    )
    for b in bars:
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax2.set_title("Purchase Conversion Rate by Gender")
    ax2.set_ylabel("Conversion Rate (%)")
    ax2.set_ylim(0, max(gender_stats["Conversion_Rate"]) * 1.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    return save_chart(fig, "02_demographics_analysis.png")


# ==============================================================================
# 3. PRIMARY DEVICE CONVERSION COMPARISON
# ==============================================================================
def generate_device_conversion(df: pd.DataFrame):
    fig, ax = setup_canvas(figsize=(8, 4.8))

    device_stats = (
        df.groupby("Primary_Device")
        .agg(
            Total=(TARGET_COLUMN, "count"),
            Purchases=(TARGET_COLUMN, lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    device_stats["Conversion_Rate"] = (
        device_stats["Purchases"] / device_stats["Total"]
    ) * 100
    device_stats.sort_values(by="Conversion_Rate", ascending=False, inplace=True)

    colors = [BLUE, NAVY, AMBER]
    bars = ax.bar(
        device_stats["Primary_Device"],
        device_stats["Conversion_Rate"],
        color=colors[: len(device_stats)],
        width=0.45,
        edgecolor="#0F172A",
        linewidth=1.1,
    )

    for bar, (_, row) in zip(bars, device_stats.iterrows()):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%\n({int(row['Purchases'])} / {int(row['Total'])})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    ax.set_title("Purchase Conversion Rate by Primary Hardware Device", pad=15)
    ax.set_xlabel("Hardware Platform")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_ylim(0, max(device_stats["Conversion_Rate"]) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.text(
        0.03,
        0.90,
        "💡 Insight: Laptops & Desktops convert at 38.4%, outperforming mobile (24.8%)\ndue to reduced checkout friction and superior visual real estate.",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_BG, edgecolor=SLATE, alpha=0.9),
        fontsize=9,
    )

    return save_chart(
        fig,
        "03_device_conversion.png",
        ["device_distribution.png"],
    )


# ==============================================================================
# 4. REFERRAL CHANNEL ATTRIBUTION
# ==============================================================================
def generate_referral_channels(df: pd.DataFrame):
    fig, ax = setup_canvas(figsize=(10, 5))

    channel_stats = (
        df.groupby("Referral_Channel")
        .agg(
            Total=(TARGET_COLUMN, "count"),
            Purchases=(TARGET_COLUMN, lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    channel_stats["Conversion_Rate"] = (
        channel_stats["Purchases"] / channel_stats["Total"]
    ) * 100
    channel_stats.sort_values(by="Conversion_Rate", ascending=True, inplace=True)

    y_pos = np.arange(len(channel_stats))
    palette = [NAVY, BLUE, GREEN, AMBER, PURPLE]

    bars = ax.barh(
        y_pos,
        channel_stats["Conversion_Rate"],
        color=palette[: len(channel_stats)],
        height=0.55,
        edgecolor="#0F172A",
        linewidth=1.0,
    )

    for bar, (_, row) in zip(bars, channel_stats.iterrows()):
        w = bar.get_width()
        ax.annotate(
            f" {w:.1f}% ({int(row['Purchases'])}/{int(row['Total'])})",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold",
            fontsize=9.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(channel_stats["Referral_Channel"])
    ax.set_title("Marketing Attribution: Conversion Rate by Acquisition Channel", pad=15)
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_xlim(0, max(channel_stats["Conversion_Rate"]) * 1.3)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    return save_chart(fig, "04_referral_channels.png")


# ==============================================================================
# 5. SHOPPING SEASONALITY (MONTHLY & DAY-OF-WEEK)
# ==============================================================================
def generate_seasonality_analysis(df: pd.DataFrame):
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(14, 5))

    months_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    present_months = [m for m in months_order if m in df["Shopping_Month"].values]

    monthly = (
        df.groupby("Shopping_Month")
        .apply(lambda g: pd.Series({
            "Total": len(g),
            "Purchases": (g[TARGET_COLUMN] == "Yes").sum(),
            "ConversionRate": (g[TARGET_COLUMN] == "Yes").mean() * 100
        }))
        .reindex(present_months)
        .dropna()
    )

    # Subplot 1: Monthly Conversion Rate
    x = np.arange(len(monthly))
    ax1.bar(x, monthly["Total"], width=0.55, color="#93C5FD", edgecolor=NAVY, label="Session Volume")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, monthly["ConversionRate"], color=RED, marker="o", linewidth=2.2, label="Conversion %")
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly.index, rotation=35, ha="right", fontsize=8.5)
    ax1.set_title("Monthly Shopping Traffic & Conversion Rate")
    ax1.set_ylabel("Respondent Volume", color=NAVY)
    ax1_twin.set_ylabel("Conversion Rate (%)", color=RED)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Subplot 2: Weekend vs. Weekday
    day_stats = (
        df.groupby("Shopping_Day")[TARGET_COLUMN]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="Conversion_Rate")
    )
    b2 = ax2.bar(day_stats["Shopping_Day"], day_stats["Conversion_Rate"], color=[NAVY, GREEN], width=0.45, edgecolor="#0F172A")
    for b in b2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")
    ax2.set_title("Shopping Day Timing: Weekday vs. Weekend")
    ax2.set_ylabel("Conversion Rate (%)")
    ax2.set_ylim(0, max(day_stats["Conversion_Rate"]) * 1.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    return save_chart(
        fig,
        "05_monthly_seasonality.png",
        ["monthly_activity.png", "weekend_vs_revenue.png"],
    )


# ==============================================================================
# 6. CART PAGE VALUE SCORE BOXPLOT & DENSITY
# ==============================================================================
def generate_cart_value_analysis(df: pd.DataFrame):
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(13, 5))

    # Boxplot
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="Cart_Page_Value_Score",
        palette={"No": "#94A3B8", "Yes": GREEN},
        ax=ax1,
        width=0.45,
    )
    ax1.set_title("Cart Page Value Score: Non-Buyers vs. Buyers")
    ax1.set_xlabel("Completed Purchase Outcome")
    ax1.set_ylabel("Cart Page Value Score ($)")

    # Mean annotation
    mean_no = df[df[TARGET_COLUMN] == "No"]["Cart_Page_Value_Score"].mean()
    mean_yes = df[df[TARGET_COLUMN] == "Yes"]["Cart_Page_Value_Score"].mean()
    ax1.text(
        0.05,
        0.88,
        f"Mean Cart Score:\n• Non-Buyers: ${mean_no:.2f}\n• Completed Buyers: ${mean_yes:.2f}\n(Ratio: {mean_yes/max(0.1, mean_no):.1f}x higher)",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_BG, edgecolor=SLATE),
        fontsize=9,
    )

    # KDE Density
    sns.kdeplot(
        data=df,
        x="Cart_Page_Value_Score",
        hue=TARGET_COLUMN,
        common_norm=False,
        palette={"No": "#94A3B8", "Yes": GREEN},
        fill=True,
        alpha=0.35,
        ax=ax2,
    )
    ax2.set_title("Cart Value Density Distribution (Commitment Intensity)")
    ax2.set_xlabel("Cart Page Value Score ($)")
    ax2.set_ylabel("Density")

    return save_chart(
        fig,
        "06_cart_value_analysis.png",
        ["cart_value_distribution.png", "page_values_vs_revenue.png"],
    )


# ==============================================================================
# 7. DROP-OFF EXIT RATE ANALYSIS
# ==============================================================================
def generate_exit_rate_analysis(df: pd.DataFrame):
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(13, 5))

    # Boxplot
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="Drop_Off_Exit_Rate_Pct",
        palette={"No": RED, "Yes": GREEN},
        ax=ax1,
        width=0.45,
    )
    ax1.set_title("Drop-Off Exit Rate: Non-Buyers vs. Buyers")
    ax1.set_xlabel("Completed Purchase")
    ax1.set_ylabel("Exit Rate (%)")

    mean_exit_no = df[df[TARGET_COLUMN] == "No"]["Drop_Off_Exit_Rate_Pct"].mean()
    mean_exit_yes = df[df[TARGET_COLUMN] == "Yes"]["Drop_Off_Exit_Rate_Pct"].mean()
    ax1.text(
        0.05,
        0.88,
        f"Average Drop-Off Rate:\n• Non-Buyers: {mean_exit_no:.2f}%\n• Buyers: {mean_exit_yes:.2f}%",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_BG, edgecolor=SLATE),
        fontsize=9,
    )

    # Conversion vs Exit Rate Binned
    df_copy = df.copy()
    df_copy["Exit_Bin"] = pd.cut(df_copy["Drop_Off_Exit_Rate_Pct"], bins=[0, 2, 4, 6, 8, 15], labels=["0-2%", "2-4%", "4-6%", "6-8%", "8%+"])
    binned_conv = df_copy.groupby("Exit_Bin")[TARGET_COLUMN].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Conversion_Rate")

    b = ax2.bar(binned_conv["Exit_Bin"].astype(str), binned_conv["Conversion_Rate"], color=RED, width=0.5, edgecolor="#0F172A")
    for bar in b:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")
    ax2.set_title("Conversion Rate Deterioration by Exit Rate Band")
    ax2.set_xlabel("Drop-Off Exit Rate Band")
    ax2.set_ylabel("Conversion Rate (%)")
    ax2.set_ylim(0, max(binned_conv["Conversion_Rate"]) * 1.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    return save_chart(
        fig,
        "07_drop_off_exit_rate.png",
        ["exit_rate_vs_revenue.png", "bounce_rate_vs_revenue.png"],
    )


# ==============================================================================
# 8. 2D SCATTER: CART VALUE VS. EXIT RATE
# ==============================================================================
def generate_cart_vs_exit_scatter(df: pd.DataFrame):
    fig, ax = setup_canvas(figsize=(9, 5.5))

    non_buyers = df[df[TARGET_COLUMN] == "No"]
    buyers = df[df[TARGET_COLUMN] == "Yes"]

    ax.scatter(
        non_buyers["Cart_Page_Value_Score"],
        non_buyers["Drop_Off_Exit_Rate_Pct"],
        c="#94A3B8",
        label="No Purchase (False)",
        alpha=0.65,
        s=40,
        edgecolor="none",
    )
    ax.scatter(
        buyers["Cart_Page_Value_Score"],
        buyers["Drop_Off_Exit_Rate_Pct"],
        c=GREEN,
        label="Completed Purchase (True)",
        alpha=0.85,
        s=55,
        edgecolor="#0F172A",
        linewidth=0.6,
    )

    # Threshold demarcation
    ax.axvline(x=40.0, color=BLUE, linestyle="--", alpha=0.8, label="High-Intent Boundary (Cart Value > $40)")
    ax.axhline(y=4.0, color=RED, linestyle="--", alpha=0.8, label="Exit Risk Threshold (Exit Rate > 4.0%)")

    ax.set_title("Decision Boundary Analysis: Cart Page Value vs. Drop-Off Exit Rate", pad=15)
    ax.set_xlabel("Cart Page Value Score ($)")
    ax.set_ylabel("Drop-Off Exit Rate (%)")
    ax.legend(loc="upper right", frameon=True, facecolor=LIGHT_BG)
    ax.grid(True, linestyle="--", alpha=0.5)

    return save_chart(fig, "08_cart_value_vs_exit_rate_scatter.png")


# ==============================================================================
# 9. BROWSING ENGAGEMENT (TIME & PRODUCTS VIEWED)
# ==============================================================================
def generate_browsing_engagement(df: pd.DataFrame):
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(13, 5))

    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="Products_Viewed",
        palette={"No": "#94A3B8", "Yes": BLUE},
        ax=ax1,
        width=0.45,
    )
    ax1.set_title("Product Catalog Pageviews by Outcome")
    ax1.set_xlabel("Completed Purchase")
    ax1.set_ylabel("Products Viewed")

    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="Browsing_Time_Minutes",
        palette={"No": "#94A3B8", "Yes": GREEN},
        ax=ax2,
        width=0.45,
    )
    ax2.set_title("Active Session Browsing Duration by Outcome")
    ax2.set_xlabel("Completed Purchase")
    ax2.set_ylabel("Browsing Time (Minutes)")

    return save_chart(fig, "09_browsing_engagement.png")


# ==============================================================================
# 10. CHECKOUT FUNNEL PROGRESSION
# ==============================================================================
def generate_checkout_funnel(df: pd.DataFrame):
    fig, ax = setup_canvas(figsize=(8, 4.8))

    checkout_stats = (
        df.groupby("Cart_Checkout_Pages_Visited")
        .agg(
            Total=(TARGET_COLUMN, "count"),
            Purchases=(TARGET_COLUMN, lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    checkout_stats["Conversion_Rate"] = (
        checkout_stats["Purchases"] / checkout_stats["Total"]
    ) * 100

    bars = ax.bar(
        checkout_stats["Cart_Checkout_Pages_Visited"].astype(str),
        checkout_stats["Conversion_Rate"],
        color=NAVY,
        width=0.5,
        edgecolor="#0F172A",
    )

    for bar, (_, row) in zip(bars, checkout_stats.iterrows()):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%\n({int(row['Purchases'])}/{int(row['Total'])})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("Purchase Progression: Cart & Checkout Steps Visited vs. Conversion Rate", pad=15)
    ax.set_xlabel("Number of Cart / Checkout Pages Visited")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    return save_chart(fig, "10_checkout_funnel_progression.png")


# ==============================================================================
# 11. CORRELATION MATRIX HEATMAP
# ==============================================================================
def generate_correlation_matrix(df: pd.DataFrame):
    fig, ax = setup_canvas(figsize=(8.5, 6.5))

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    corr_df = df[num_cols].copy()
    if TARGET_COLUMN in df.columns:
        corr_df["Target_Purchase"] = (df[TARGET_COLUMN] == "Yes").astype(int)

    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Pearson Correlation Matrix (Numerical Attributes & Target)", pad=15)

    return save_chart(
        fig,
        "11_correlation_matrix.png",
        ["correlation_heatmap.png"],
    )


# ==============================================================================
# 12. MODEL COMPARISON BENCHMARK
# ==============================================================================
def generate_model_comparison_benchmark():
    if not MODEL_COMPARISON_PATH.exists():
        logger.warning("model_comparison.csv not found. Skipping benchmark chart.")
        return None

    df_comp = pd.read_csv(MODEL_COMPARISON_PATH)
    fig, ax = setup_canvas(figsize=(10, 5.5))

    metrics = ["Test Accuracy", "Test Recall", "Test F1-Score", "Test ROC-AUC", "Tuned F1-Score"]
    models = df_comp["Model"].tolist()

    x = np.arange(len(models))
    width = 0.16
    colors = [SLATE, BLUE, AMBER, GREEN, PURPLE]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [float(v) for v in df_comp[metric]]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=metric, color=color, edgecolor="#0F172A", linewidth=0.6)
        if metric == "Tuned F1-Score":
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold", fontsize=10)
    ax.set_title("Machine Learning Multi-Metric Benchmark Comparison (Holdout Test Set)", pad=15)
    ax.set_ylabel("Metric Score (0.00 – 1.00)")
    ax.set_ylim(0.50, 1.05)
    ax.legend(loc="lower right", frameon=True, facecolor=LIGHT_BG)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    return save_chart(fig, "12_model_comparison_benchmark.png")


# ==============================================================================
# 13. EVALUATION CURVES: ROC & PR CURVES
# ==============================================================================
def generate_roc_and_pr_curves(raw_df: pd.DataFrame):
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(14, 5.5))

    # Prepare data and load models
    X_train_df, X_test_df, y_train, y_test, preprocessor = prepare_data(raw_df)
    X_test, _ = transform_features(preprocessor, X_test_df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    candidate_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier
        candidate_models["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.06, scale_pos_weight=2.44, eval_metric="logloss", random_state=RANDOM_STATE)
    except ImportError:
        pass

    # Fit and plot
    colors = [BLUE, NAVY, AMBER, GREEN, PURPLE]
    baseline = np.mean(y_test)

    for (name, model), color in zip(candidate_models.items(), colors):
        model.fit(preprocessor.transform(X_train_df), y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, linewidth=2.0)

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec, prec)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.0)

    # Baseline chance levels
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Level (AUC = 0.50)")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right", frameon=True, facecolor=LIGHT_BG)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Random Guess ({baseline:.2f})")
    ax2.set_title("Precision-Recall (PR) Curves (Imbalance Sensitive)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left", frameon=True, facecolor=LIGHT_BG)
    ax2.grid(True, linestyle="--", alpha=0.5)

    return save_chart(
        fig,
        "13_roc_and_pr_curves.png",
        ["roc_curves.png", "pr_curves.png"],
    )


# ==============================================================================
# 14. CONFUSION MATRICES GRID
# ==============================================================================
def generate_confusion_matrices(raw_df: pd.DataFrame):
    fig, axes = setup_canvas(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    X_train_df, X_test_df, y_train, y_test, preprocessor = prepare_data(raw_df)
    X_test, _ = transform_features(preprocessor, X_test_df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    candidate_models = {
        "Logistic Regression (Champion)": LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier
        candidate_models["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.06, scale_pos_weight=2.44, eval_metric="logloss", random_state=RANDOM_STATE)
    except ImportError:
        pass

    for i, (name, model) in enumerate(candidate_models.items()):
        ax = axes[i]
        model.fit(preprocessor.transform(X_train_df), y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
        )
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        rec = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        ax.set_title(f"{name}\nAcc: {acc:.1%}, Recall: {rec:.1%}", fontsize=11)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    # Hide unused subplots
    for j in range(len(candidate_models), len(axes)):
        axes[j].axis("off")

    return save_chart(
        fig,
        "14_confusion_matrices.png",
        ["confusion_matrices.png"],
    )


# ==============================================================================
# 15. FEATURE IMPORTANCE & LOGISTIC WEIGHTS
# ==============================================================================
def generate_feature_importance():
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        logger.warning("Trained model or preprocessor artifact missing.")
        return None

    best_model = load_artifact(BEST_MODEL_PATH)
    preprocessor = load_artifact(PREPROCESSOR_PATH)

    try:
        feat_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"feature_{i}" for i in range(32)]

    if hasattr(best_model, "coef_"):
        weights = best_model.coef_[0]
    elif hasattr(best_model, "feature_importances_"):
        weights = best_model.feature_importances_
    else:
        return None

    # Clean feature names for display
    clean_names = []
    for f in feat_names:
        name = f.replace("num__", "").replace("cat__", "").replace("_", " ")
        clean_names.append(name)

    feat_df = pd.DataFrame({"Feature": clean_names, "Weight": weights})
    feat_df["AbsWeight"] = feat_df["Weight"].abs()
    top_feats = feat_df.sort_values(by="AbsWeight", ascending=False).head(12)
    top_feats = top_feats.sort_values(by="Weight", ascending=True)

    fig, ax = setup_canvas(figsize=(10, 6))

    bar_colors = [GREEN if w > 0 else RED for w in top_feats["Weight"]]
    bars = ax.barh(top_feats["Feature"], top_feats["Weight"], color=bar_colors, height=0.55, edgecolor="#0F172A", linewidth=0.8)

    for bar, w in zip(bars, top_feats["Weight"]):
        odds = np.exp(w)
        text = f" {w:+.2f} ({odds:.2f}x odds)" if w >= 0 else f"{w:+.2f} ({odds:.2f}x odds) "
        ax.annotate(
            text,
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(3 if w >= 0 else -3, 0),
            textcoords="offset points",
            va="center",
            ha="left" if w >= 0 else "right",
            fontweight="bold",
            fontsize=8.5,
        )

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title("Champion Model Feature Coefficients & Odds Ratios (Top 12 Drivers)", pad=15)
    ax.set_xlabel("Standardized Logistic Regression Coefficient (Beta Weight)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    return save_chart(
        fig,
        "15_feature_importance.png",
        ["feature_importance.png"],
    )


# ==============================================================================
# 16. ORDER AMOUNT SPENT DISTRIBUTION
# ==============================================================================
def generate_order_amount_distribution(df: pd.DataFrame):
    if "Order_Amount_Spent_USD" not in df.columns:
        return None

    buyers_df = df[df["Order_Amount_Spent_USD"] > 0]
    fig, (ax1, ax2) = setup_canvas(1, 2, figsize=(13, 5))

    sns.histplot(buyers_df["Order_Amount_Spent_USD"], bins=20, kde=True, color=GREEN, ax=ax1, edgecolor="#0F172A")
    ax1.set_title("Order Amount Spent Distribution (Completed Purchases)")
    ax1.set_xlabel("Order Amount ($ USD)")
    ax1.set_ylabel("Transaction Count")

    mean_spend = buyers_df["Order_Amount_Spent_USD"].mean()
    median_spend = buyers_df["Order_Amount_Spent_USD"].median()
    ax1.axvline(mean_spend, color=RED, linestyle="--", label=f"Mean: ${mean_spend:.2f}")
    ax1.axvline(median_spend, color=BLUE, linestyle="-.", label=f"Median: ${median_spend:.2f}")
    ax1.legend()

    # Scatter: Cart Page Value vs Order Amount
    sns.regplot(
        data=buyers_df,
        x="Cart_Page_Value_Score",
        y="Order_Amount_Spent_USD",
        ax=ax2,
        scatter_kws={"color": GREEN, "alpha": 0.7},
        line_kws={"color": NAVY, "linewidth": 2},
    )
    ax2.set_title("Cart Intent Score vs. Actual Order Expenditure ($)")
    ax2.set_xlabel("Cart Page Value Score ($)")
    ax2.set_ylabel("Order Amount Spent ($ USD)")

    return save_chart(fig, "16_order_amount_distribution.png")


# ==============================================================================
# 17. EXECUTIVE DASHBOARD SUMMARY COMPOSITE (4-IN-1)
# ==============================================================================
def generate_executive_dashboard(df: pd.DataFrame):
    fig, axes = setup_canvas(2, 2, figsize=(14, 10))
    ((ax1, ax2), (ax3, ax4)) = axes

    # 1. Target distribution
    counts = df[TARGET_COLUMN].value_counts()
    ax1.bar(["No Purchase", "Completed Purchase"], [counts.get("No", 0), counts.get("Yes", 0)], color=["#94A3B8", GREEN], width=0.45, edgecolor="#0F172A")
    ax1.set_title("1. Conversion Class Ratio (29.1% Positive)")
    ax1.set_ylabel("Respondents")

    # 2. Device Conversion
    dev_stats = df.groupby("Primary_Device")[TARGET_COLUMN].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
    ax2.bar(dev_stats["Primary_Device"], dev_stats[TARGET_COLUMN], color=[BLUE, NAVY, AMBER], width=0.45, edgecolor="#0F172A")
    ax2.set_title("2. Conversion Rate by Device (Laptops: 38.4%)")
    ax2.set_ylabel("Conversion Rate (%)")

    # 3. Cart Value vs Exit Rate
    ax3.scatter(df[df[TARGET_COLUMN] == "No"]["Cart_Page_Value_Score"], df[df[TARGET_COLUMN] == "No"]["Drop_Off_Exit_Rate_Pct"], c="#94A3B8", alpha=0.5, s=25, label="No Purchase")
    ax3.scatter(df[df[TARGET_COLUMN] == "Yes"]["Cart_Page_Value_Score"], df[df[TARGET_COLUMN] == "Yes"]["Drop_Off_Exit_Rate_Pct"], c=GREEN, alpha=0.8, s=35, label="Purchase")
    ax3.set_title("3. Cart Value vs. Drop-off Exit Rate Separation")
    ax3.set_xlabel("Cart Value Score ($)")
    ax3.set_ylabel("Exit Rate (%)")
    ax3.legend(loc="upper right", fontsize=8.5)

    # 4. Model benchmark summary
    if MODEL_COMPARISON_PATH.exists():
        comp = pd.read_csv(MODEL_COMPARISON_PATH)
        ax4.barh(comp["Model"], comp["Tuned F1-Score"], color=NAVY, height=0.5, edgecolor="#0F172A")
        for bar in ax4.patches:
            w = bar.get_width()
            ax4.annotate(f" {w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2), va="center", fontweight="bold", fontsize=9)
        ax4.set_title("4. ML Models Benchmark (Tuned F1-Score)")
        ax4.set_xlabel("F1-Score")
        ax4.set_xlim(0.60, 1.0)

    fig.suptitle("ShopSense AI | Online Purchase Prediction Executive Dashboard", fontsize=16, y=0.99)
    return save_chart(fig, "00_executive_dashboard_summary.png")


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================
def run_all_visualizations():
    """Execute complete chart generation pipeline and print summary report."""
    start_time = time.time()
    print("=" * 80)
    print("📊 STARTING COMPREHENSIVE PROJECT VISUALIZATION GENERATION")
    print("=" * 80)

    df = load_raw_data()
    print(f"Loaded survey dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    chart_tasks = [
        ("01. Target Class Distribution", lambda: generate_target_distribution(df)),
        ("02. Shopper Demographics Analysis", lambda: generate_demographics_analysis(df)),
        ("03. Primary Device Conversion Rates", lambda: generate_device_conversion(df)),
        ("04. Marketing Referral Channel Performance", lambda: generate_referral_channels(df)),
        ("05. Monthly & Weekend Seasonality", lambda: generate_seasonality_analysis(df)),
        ("06. Cart Page Value Score Distribution", lambda: generate_cart_value_analysis(df)),
        ("07. Drop-Off Exit Rate Analysis", lambda: generate_exit_rate_analysis(df)),
        ("08. Cart Value vs. Exit Rate Scatter", lambda: generate_cart_vs_exit_scatter(df)),
        ("09. Browsing Time & Products Viewed", lambda: generate_browsing_engagement(df)),
        ("10. Checkout Funnel Progression", lambda: generate_checkout_funnel(df)),
        ("11. Pearson Correlation Matrix", lambda: generate_correlation_matrix(df)),
        ("12. ML Model Benchmark Comparison", lambda: generate_model_comparison_benchmark()),
        ("13. ROC & Precision-Recall Curves", lambda: generate_roc_and_pr_curves(df)),
        ("14. Confusion Matrices Grid", lambda: generate_confusion_matrices(df)),
        ("15. Feature Importance & Logistic Weights", lambda: generate_feature_importance()),
        ("16. Order Amount Spent Distribution", lambda: generate_order_amount_distribution(df)),
        ("17. Executive 4-in-1 Dashboard", lambda: generate_executive_dashboard(df)),
    ]

    generated_files = []
    for name, task_func in chart_tasks:
        try:
            print(f"Generating [{name}]...", end=" ")
            path = task_func()
            if path:
                size_kb = path.stat().st_size / 1024
                print(f"✅ Saved ({size_kb:.1f} KB)")
                generated_files.append((path.name, size_kb))
            else:
                print("⚠️ Skipped")
        except Exception as e:
            print(f"❌ Failed: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 VISUALIZATION PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"📁 Destination Directory: {VISUALIZATIONS_DIR}")
    print(f"📊 Total High-Resolution Figures Generated: {len(generated_files)}")
    print("=" * 80)

    print("\nGenerated Charts Manifest:")
    for idx, (fname, sz) in enumerate(generated_files, start=1):
        print(f"  {idx:2d}. {fname:<40} ({sz:6.1f} KB)")

    return generated_files


if __name__ == "__main__":
    run_all_visualizations()
