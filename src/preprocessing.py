"""
Data Preprocessing Pipeline Module.
Implements missing value handling, feature separation, ColumnTransformer with
StandardScaler and OneHotEncoder, and stratified train-test splitting.
Designed for the human-survey online shopping dataset.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils import logger, save_artifact


def create_preprocessor() -> ColumnTransformer:
    """
    Construct Scikit-Learn ColumnTransformer for numerical and categorical pipelines.
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Clean survey dataset:
    - Handle duplicates (excluding Respondent_ID)
    - Ensure correct data types for categorical features
    - Map target variable ('Yes' / 'No' or True / False) to integer binary (1 / 0)
    """
    df_clean = df.copy()

    if drop_duplicates:
        initial_count = len(df_clean)
        # Check duplicates on feature columns
        feature_subset = [c for c in ALL_FEATURE_COLUMNS if c in df_clean.columns]
        if feature_subset:
            df_clean = df_clean.drop_duplicates(subset=feature_subset)
        else:
            df_clean = df_clean.drop_duplicates()
        dropped = initial_count - len(df_clean)
        logger.info(f"Removed {dropped} duplicate records. Remaining: {len(df_clean)} rows.")

    # Cast categorical features to string
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)

    # Cast numerical features to float
    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Map target to binary integer (0 or 1)
    if TARGET_COLUMN in df_clean.columns:
        if df_clean[TARGET_COLUMN].dtype == object or df_clean[TARGET_COLUMN].dtype == bool:
            target_map = {
                "Yes": 1, "No": 0, "yes": 1, "no": 0,
                True: 1, False: 0, "True": 1, "False": 0,
                "1": 1, "0": 0, 1: 1, 0: 0
            }
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map(target_map).fillna(0).astype(int)
        else:
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(int)

    return df_clean


def prepare_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Execute full data preparation workflow:
    1. Clean data and cast types
    2. Split features (X) and target (y)
    3. Perform stratified train-test split
    4. Fit ColumnTransformer preprocessor on training data only (preventing data leakage)
    5. Save preprocessor artifact

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df_cleaned = clean_dataset(df, drop_duplicates=drop_duplicates)

    X = df_cleaned[ALL_FEATURE_COLUMNS].copy()
    y = df_cleaned[TARGET_COLUMN].copy()

    # Stratified train-test split to preserve target ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Data split: Train={X_train.shape[0]} samples ({y_train.mean():.2%} positive), "
        f"Test={X_test.shape[0]} samples ({y_test.mean():.2%} positive)"
    )

    # Initialize and fit preprocessor ONLY on X_train
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    logger.info("Fitted ColumnTransformer preprocessor on X_train (data leakage prevented).")

    # Save fitted preprocessor
    save_artifact(preprocessor, PREPROCESSOR_PATH)

    return X_train, X_test, y_train, y_test, preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Transform input dataframe with fitted preprocessor and retrieve feature names.
    """
    X_transformed = preprocessor.transform(X)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    return X_transformed, feature_names


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw_df = load_raw_data()
    X_train, X_test, y_train, y_test, prep = prepare_data(raw_df)
    X_train_trans, names = transform_features(prep, X_train)
    print("Preprocessing verification:")
    print(f"X_train transformed shape: {X_train_trans.shape}")
    print(f"Transformed features count: {len(names)}")
    print("Transformed feature names:", names)
