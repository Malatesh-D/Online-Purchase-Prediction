"""
Prediction Pipeline Module.
Loads preprocessor and best-performing model to perform real-time purchase intention predictions.
Outputs class predictions, probabilities, confidence metrics, estimated order values, and domain interpretations.
Tuned specifically for the human-survey online shopping dataset.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ALL_FEATURE_COLUMNS,
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
)
from src.utils import load_artifact, load_json, logger


class PurchasePredictor:
    """Production predictor encapsulating trained model and preprocessing pipeline."""

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metadata_path = metadata_path
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.optimal_threshold = 0.5
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model, preprocessor, and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Run train.py first.")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {self.preprocessor_path}. Run train.py first.")

        self.model = load_artifact(self.model_path)
        self.preprocessor = load_artifact(self.preprocessor_path)

        if self.metadata_path.exists():
            self.metadata = load_json(self.metadata_path)
            self.optimal_threshold = float(self.metadata.get("optimal_threshold", 0.5))
        else:
            self.optimal_threshold = 0.5

        logger.info(
            f"PurchasePredictor initialized with model '{getattr(self.model, '__class__', type(self.model)).__name__}' "
            f"and threshold {self.optimal_threshold:.2f}"
        )

    def validate_and_format_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Validate input survey schema, ensure all 12 expected features are present, and format datatypes.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Input data must be a dictionary or pandas DataFrame, got {type(data)}")

        missing_cols = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        df = df[ALL_FEATURE_COLUMNS].copy()

        # Format numerical types
        for col in NUMERICAL_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Format categorical types to strings
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype(str)

        return df

    def predict(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
        use_optimal_threshold: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate purchase intention prediction for an input respondent / shopper session.

        Returns:
            Dictionary containing:
            - predicted_class: 'Likely to Purchase' or 'Unlikely to Purchase'
            - raw_class: 1 or 0
            - purchase_probability: float (0.0 to 1.0)
            - non_purchase_probability: float (0.0 to 1.0)
            - purchase_probability_pct: formatted string (e.g. "86.4%")
            - non_purchase_probability_pct: formatted string
            - confidence_score: float (percentage)
            - threshold_used: float
            - estimated_order_amount: float (estimated cart value in USD if converted)
            - interpretation: academic and business interpretation text
            - risk_level: 'High Intent Shopper', 'Moderate Intent', 'Window Shopper', or 'High Exit / Inactive'
        """
        df_formatted = self.validate_and_format_input(data)

        # Transform features through pipeline
        X_trans = self.preprocessor.transform(df_formatted)

        # Compute probabilities from trained model
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_trans)[0]
            prob_neg = float(probas[0])
            prob_pos = float(probas[1])
        else:
            raw_pred = self.model.predict(X_trans)[0]
            prob_pos = float(raw_pred)
            prob_neg = 1.0 - prob_pos

        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        is_purchase = bool(prob_pos >= threshold)

        confidence = max(prob_pos, prob_neg) * 100

        # Features for contextual explanation
        age = int(df_formatted["Age"].iloc[0])
        device = str(df_formatted["Primary_Device"].iloc[0])
        cart_value = float(df_formatted["Cart_Page_Value_Score"].iloc[0])
        checkout_pages = int(df_formatted["Cart_Checkout_Pages_Visited"].iloc[0])
        exit_rate = float(df_formatted["Drop_Off_Exit_Rate_Pct"].iloc[0])
        channel = str(df_formatted["Referral_Channel"].iloc[0])

        # Estimate order value if converted
        if is_purchase:
            if cart_value > 0:
                est_order = round(max(25.0, cart_value * 1.25 + 15.0), 2)
            else:
                est_order = round(float(np.random.uniform(35.0, 75.0)), 2)
        else:
            est_order = 0.0

        if prob_pos >= 0.75:
            risk_level = "High Intent Shopper"
            interpretation = (
                f"Respondent ({age} yrs, browsing on {device} via {channel}) shows very strong commercial commitment "
                f"({prob_pos * 100:.1f}% purchase probability). The high cart value score (${cart_value:.2f}) and multiple checkout "
                f"steps ({checkout_pages}) indicate ready-to-buy intent. Recommendation: Provide 1-click checkout and minimal form friction."
            )
        elif prob_pos >= threshold:
            risk_level = "Moderate Intent (Nudgeable)"
            interpretation = (
                f"Respondent ({age} yrs, on {device}) shows moderate purchasing intent ({prob_pos * 100:.1f}% probability). "
                f"Cart engagement is positive (${cart_value:.2f}), but exit risk is {exit_rate:.1f}%. "
                "Recommendation: Trigger a gentle real-time nudge such as free standard shipping or an instant 5% promo code."
            )
        elif prob_pos >= 0.25:
            risk_level = "Window Shopper"
            interpretation = (
                f"Respondent is primarily exploring the catalog ({prob_pos * 100:.1f}% purchase likelihood). "
                f"Exit drop-off is {exit_rate:.1f}% with limited checkout progression ({checkout_pages} pages). "
                "Recommendation: Encourage wishlist creation, product bookmarking, or email newsletter sign-up."
            )
        else:
            risk_level = "High Exit / Inactive"
            interpretation = (
                f"Respondent shows very low conversion probability ({prob_pos * 100:.1f}%). "
                f"Elevated drop-off rate ({exit_rate:.1f}%) and zero cart commitment. "
                "Recommendation: Avoid intrusive popups; prioritize site responsiveness and intuitive catalog navigation."
            )

        result = {
            "predicted_class": "Likely to Purchase" if is_purchase else "Unlikely to Purchase",
            "completed_purchase": "Yes" if is_purchase else "No",
            "raw_class": 1 if is_purchase else 0,
            "purchase_probability": round(prob_pos, 4),
            "non_purchase_probability": round(prob_neg, 4),
            "purchase_probability_pct": f"{prob_pos * 100:.1f}%",
            "non_purchase_probability_pct": f"{prob_neg * 100:.1f}%",
            "confidence_score": round(confidence, 1),
            "confidence_score_pct": f"{confidence:.1f}%",
            "threshold_used": round(threshold, 3),
            "estimated_order_amount": est_order,
            "estimated_order_amount_usd": f"${est_order:,.2f}" if is_purchase else "$0.00",
            "risk_level": risk_level,
            "interpretation": interpretation,
        }
        return result


_default_predictor = None


def get_predictor() -> PurchasePredictor:
    """Retrieve or create singleton PurchasePredictor instance."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = PurchasePredictor()
    return _default_predictor


def predict_single(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to predict purchase outcome for a single respondent dict."""
    predictor = get_predictor()
    return predictor.predict(data)


if __name__ == "__main__":
    sample_buyer = {
        "Age": 23,
        "Gender": "Male",
        "Customer_Type": "Returning Customer",
        "Primary_Device": "Laptop / Desktop",
        "Shopping_Month": "November",
        "Shopping_Day": "Weekday",
        "Referral_Channel": "Direct Website Visit",
        "Products_Viewed": 16,
        "Browsing_Time_Minutes": 17.7,
        "Cart_Checkout_Pages_Visited": 2,
        "Cart_Page_Value_Score": 80.70,
        "Drop_Off_Exit_Rate_Pct": 3.10,
    }

    sample_browser = {
        "Age": 41,
        "Gender": "Female",
        "Customer_Type": "Returning Customer",
        "Primary_Device": "Smartphone",
        "Shopping_Month": "February",
        "Shopping_Day": "Weekday",
        "Referral_Channel": "Google Search",
        "Products_Viewed": 2,
        "Browsing_Time_Minutes": 5.8,
        "Cart_Checkout_Pages_Visited": 0,
        "Cart_Page_Value_Score": 0.0,
        "Drop_Off_Exit_Rate_Pct": 5.44,
    }

    print("\n--- Testing High Intent Buyer Sample ---")
    try:
        res1 = predict_single(sample_buyer)
        for k, v in res1.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Note: Train model first before testing prediction:", e)
