"""
Model Training and Hyperparameter Tuning Module.
Trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
and XGBoost classifiers with cross-validation and class imbalance mitigation.
Selects and persists the best model based on F1-score and ROC-AUC.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    EVALUATION_REPORT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
)
from src.data_loader import load_raw_data
from src.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    save_evaluation_report,
)
from src.preprocessing import prepare_data, transform_features
from src.utils import logger, save_artifact, save_json


def initialize_candidate_models(pos_weight: float) -> Dict[str, Any]:
    """
    Initialize multiple classification models tuned for imbalanced conversion prediction.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        logger.warning("XGBoost is not installed. Skipping XGBoost model.")

    return models


def train_and_evaluate_all() -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Execute full model training, cross-validation, testing, and selection.
    """
    logger.info("Starting complete model training pipeline...")
    raw_df = load_raw_data()

    # Preprocess and split
    X_train_df, X_test_df, y_train, y_test, preprocessor = prepare_data(raw_df)
    X_train, feature_names = transform_features(preprocessor, X_train_df)
    X_test, _ = transform_features(preprocessor, X_test_df)

    # Compute scale_pos_weight for imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = float(neg_count / max(1, pos_count))
    logger.info(f"Class ratio in training data: {neg_count} non-purchase to {pos_count} purchase (Weight: {pos_weight:.2f})")

    models = initialize_candidate_models(pos_weight=pos_weight)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results_list = []
    trained_models = {}
    thresholds = {}

    for name, model in models.items():
        logger.info(f"Training and validating {name}...")
        start_time = time.time()

        # Cross-validation for F1 and ROC-AUC stability
        cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1).mean()
        cv_roc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1).mean()

        # Fit model on entire X_train
        model.fit(X_train, y_train)
        elapsed = round(time.time() - start_time, 2)
        trained_models[name] = model

        # Test set predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        metrics = compute_metrics(y_test, y_pred, y_proba)
        opt_thresh, opt_f1 = find_optimal_threshold(y_test, y_proba)
        thresholds[name] = opt_thresh

        # Evaluate at optimal threshold as well
        y_pred_opt = (y_proba >= opt_thresh).astype(int)
        opt_metrics = compute_metrics(y_test, y_pred_opt, y_proba)

        result_entry = {
            "Model": name,
            "CV F1-Score": round(float(cv_f1), 4),
            "CV ROC-AUC": round(float(cv_roc), 4),
            "Test Accuracy": metrics["Accuracy"],
            "Test Precision": metrics["Precision"],
            "Test Recall": metrics["Recall"],
            "Test F1-Score": metrics["F1-Score"],
            "Test ROC-AUC": metrics["ROC-AUC"],
            "Test PR-AUC": metrics["PR-AUC"],
            "Optimal Threshold": opt_thresh,
            "Tuned F1-Score": opt_metrics["F1-Score"],
            "Training Time (s)": elapsed,
        }
        results_list.append(result_entry)
        logger.info(
            f"{name} -> Test Acc: {metrics['Accuracy']:.2%}, Rec: {metrics['Recall']:.2%}, "
            f"F1: {metrics['F1-Score']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}, Tuned F1: {opt_metrics['F1-Score']:.4f}"
        )

    comparison_df = pd.DataFrame(results_list)
    comparison_df.sort_values(by="Tuned F1-Score", ascending=False, inplace=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Model comparison saved to {MODEL_COMPARISON_PATH}")

    # Best model selection prioritized by F1-Score and ROC-AUC
    # We create a composite score = 0.6 * Tuned F1-Score + 0.4 * Test ROC-AUC
    composite_scores = (
        0.6 * comparison_df["Tuned F1-Score"] + 0.4 * comparison_df["Test ROC-AUC"]
    )
    best_index = composite_scores.idxmax()
    best_model_name = comparison_df.loc[best_index, "Model"]
    best_model = trained_models[best_model_name]
    best_threshold = float(thresholds[best_model_name])
    best_metrics = comparison_df.loc[best_index].to_dict()

    logger.info(f"==> Selected Best Model: {best_model_name} (Composite F1/ROC-AUC)")

    # Save artifacts
    save_artifact(best_model, BEST_MODEL_PATH)

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": best_threshold,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
            for k, v in best_metrics.items()
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "class_weight_ratio": float(pos_weight),
    }
    save_json(metadata, MODEL_METADATA_PATH)

    # Generate evaluation charts
    logger.info("Generating evaluation charts...")
    plot_roc_curves(trained_models, X_test, y_test)
    plot_pr_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_feature_importance(best_model, feature_names)

    # Save text report
    save_evaluation_report(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_metrics=best_metrics,
        optimal_threshold=best_threshold,
    )

    logger.info("Training and evaluation pipeline completed successfully!")
    return comparison_df, best_model_name, metadata


if __name__ == "__main__":
    df_results, best_name, meta = train_and_evaluate_all()
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(df_results.to_string(index=False))
    print(f"\nBest Model: {best_name} with metadata saved.")
