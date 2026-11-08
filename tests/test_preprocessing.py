"""
Unit tests for data preprocessing module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALL_FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from src.data_loader import load_raw_data
from src.preprocessing import clean_dataset, create_preprocessor, prepare_data, transform_features


@pytest.fixture(scope="module")
def raw_df():
    """Load dataset fixture."""
    return load_raw_data()


def test_clean_dataset(raw_df):
    """Verify duplicate removal and type formatting."""
    cleaned = clean_dataset(raw_df, drop_duplicates=True)
    assert len(cleaned) <= len(raw_df)
    assert cleaned.duplicated().sum() == 0
    assert cleaned[TARGET_COLUMN].dtype in ["int32", "int64"]


def test_prepare_data_split(raw_df):
    """Verify 80:20 stratified split and preprocessor creation."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        raw_df, test_size=0.2, random_state=42
    )

    total_samples = len(X_train) + len(X_test)
    assert len(X_train) == int(round(total_samples * 0.8))
    # Stratification check: ratio within 1%
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    X_train_trans, names = transform_features(preprocessor, X_train)
    assert X_train_trans.shape[0] == len(X_train)
    assert len(names) == X_train_trans.shape[1]
