"""
Model Evaluation and Metrics Module.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix,
and generates comparison tables and evaluation curves.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import EVALUATION_REPORT_PATH, MODEL_COMPARISON_PATH, VISUALIZATIONS_DIR
from src.utils import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute all standard evaluation metrics for binary classification."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc_auc = 0.5

    # Compute PR-AUC (Average Precision)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC": round(float(pr_auc), 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find decision threshold that maximizes F1-score on the positive (purchase) class.
    Returns (optimal_threshold, best_f1).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return float(round(best_thresh, 3)), float(round(best_score, 4))


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = VISUALIZATIONS_DIR / "roc_curves.png",
) -> None:
    """Plot overlaid ROC curves for all models."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#2ca02c", "#ff7f0e"]

    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            linewidth=2,
            color=colors[i % len(colors)],
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved ROC curves to {save_path}")


def plot_pr_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = VISUALIZATIONS_DIR / "pr_curves.png",
) -> None:
    """Plot Precision-Recall curves (critical for imbalanced classes)."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#2ca02c", "#ff7f0e"]
    baseline = np.mean(y_test)

    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(rec, prec)
        plt.plot(
            rec,
            prec,
            label=f"{name} (PR-AUC = {pr_auc:.3f})",
            linewidth=2,
            color=colors[i % len(colors)],
        )

    plt.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        alpha=0.6,
        label=f"Baseline Prevalance ({baseline:.1%})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve Comparison")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved PR curves to {save_path}")


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = VISUALIZATIONS_DIR / "confusion_matrices.png",
) -> None:
    """Plot confusion matrix heatmaps in a grid for all evaluated models."""
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.8), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
        )
        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrices grid to {save_path}")


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Path = VISUALIZATIONS_DIR / "feature_importance.png",
) -> pd.DataFrame:
    """Extract and plot top N feature importances or coefficients."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        metric_name = "Importance Score"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        metric_name = "Absolute Coefficient"
    else:
        logger.warning("Model does not expose feature_importances_ or coef_")
        return pd.DataFrame()

    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .head(top_n)
    )

    # Clean display names: remove 'num__' and 'cat__' prefixes
    fi_df["CleanFeature"] = (
        fi_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    plt.figure(figsize=(9, 6), dpi=300)
    sns.barplot(
        data=fi_df,
        y="CleanFeature",
        x="Importance",
        hue="CleanFeature",
        palette="crest",
        legend=False,
    )
    plt.title(f"Top {top_n} Most Influential Features ({metric_name})")
    plt.xlabel(metric_name)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")
    return fi_df


def save_evaluation_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_metrics: Dict[str, float],
    optimal_threshold: float,
    save_path: Path = EVALUATION_REPORT_PATH,
) -> None:
    """Generate and write comprehensive evaluation report text."""
    report_lines = [
        "=" * 70,
        "ONLINE SHOPPING PURCHASE INTENTION PREDICTION - MODEL EVALUATION REPORT",
        "=" * 70,
        "",
        "1. EXECUTIVE SUMMARY:",
        f"   - Target Variable: Revenue (Binary: Purchase=1, No Purchase=0)",
        f"   - Best Performing Model: {best_model_name}",
        f"   - Best F1-Score: {best_metrics.get('F1-Score', 0):.4f}",
        f"   - Best ROC-AUC: {best_metrics.get('ROC-AUC', 0):.4f}",
        f"   - Optimal Decision Threshold: {optimal_threshold:.2f}",
        "",
        "2. COMPLETE MODEL COMPARISON BENCHMARK:",
        comparison_df.to_string(index=False),
        "",
        "3. MODEL SELECTION RATIONALE:",
        "   - The dataset exhibits an 85:15 negative:positive class imbalance.",
        "   - Naive accuracy is deceptive (a dummy model predicting all negatives achieves ~84.5% accuracy).",
        f"   - {best_model_name} demonstrated the highest combined F1-Score and ROC-AUC,",
        "     balancing high recall (capturing potential buyers) with acceptable precision (minimizing wasted marketing outreach).",
        "",
        "4. KEY BUSINESS METRICS:",
        f"   - Recall: {best_metrics.get('Recall', 0):.2%}",
        f"   - Precision: {best_metrics.get('Precision', 0):.2%}",
        f"   - Accuracy: {best_metrics.get('Accuracy', 0):.2%}",
        f"   - PR-AUC: {best_metrics.get('PR-AUC', 0):.4f}",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved evaluation text report to {save_path}")
