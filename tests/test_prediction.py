"""
Unit tests for the prediction pipeline module.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import PurchasePredictor, get_predictor


@pytest.fixture(scope="module")
def predictor():
    """Predictor fixture."""
    return get_predictor()


@pytest.fixture
def sample_shopper():
    """Valid shopping survey payload fixture."""
    return {
        "Age": 24,
        "Gender": "Female",
        "Customer_Type": "Returning Customer",
        "Primary_Device": "Smartphone",
        "Shopping_Month": "November",
        "Shopping_Day": "Weekend",
        "Referral_Channel": "Instagram / Social Ad",
        "Products_Viewed": 22,
        "Browsing_Time_Minutes": 28.5,
        "Cart_Checkout_Pages_Visited": 2,
        "Cart_Page_Value_Score": 65.0,
        "Drop_Off_Exit_Rate_Pct": 2.1,
    }


def test_predictor_initialization(predictor):
    """Verify predictor properly loaded the model and preprocessor."""
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert 0.0 <= predictor.optimal_threshold <= 1.0


def test_predict_structure_and_probabilities(predictor, sample_shopper):
    """Verify prediction result structure and probability mathematics."""
    result = predictor.predict(sample_shopper)

    expected_keys = [
        "predicted_class",
        "completed_purchase",
        "raw_class",
        "purchase_probability",
        "non_purchase_probability",
        "purchase_probability_pct",
        "non_purchase_probability_pct",
        "confidence_score",
        "estimated_order_amount",
        "estimated_order_amount_usd",
        "interpretation",
        "risk_level",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key '{key}' in prediction response"

    prob_pos = result["purchase_probability"]
    prob_neg = result["non_purchase_probability"]

    assert 0.0 <= prob_pos <= 1.0
    assert 0.0 <= prob_neg <= 1.0
    assert abs((prob_pos + prob_neg) - 1.0) < 1e-3
    assert result["predicted_class"] in ["Likely to Purchase", "Unlikely to Purchase"]
    assert len(result["interpretation"]) > 20


def test_missing_field_raises_error(predictor):
    """Verify invalid input with missing columns raises informative ValueError."""
    incomplete = {"Age": 25, "Gender": "Male"}
    with pytest.raises(ValueError) as excinfo:
        predictor.predict(incomplete)
    assert "Missing required feature columns" in str(excinfo.value)
