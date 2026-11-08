"""
Dataset acquisition and loading module.
Loads the manual shopper survey dataset (or downloads legacy benchmark if needed),
validates data integrity, and provides dataset summaries.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import DATA_DIR, DATA_FILE_PATH, TARGET_COLUMN
from src.utils import logger


def load_raw_data(data_path: Path = DATA_FILE_PATH) -> pd.DataFrame:
    """
    Load raw shopper survey dataset into a pandas DataFrame.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded survey dataset with shape {df.shape} from {data_path}")
    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect dataset characteristics:
    - Dimensions
    - Column names
    - Data types
    - Missing values
    - Duplicate count
    - Target distribution
    """
    target_counts = {}
    target_pct = {}
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        pcts = df[TARGET_COLUMN].value_counts(normalize=True) * 100
        target_counts = {str(k): int(v) for k, v in counts.items()}
        target_pct = {str(k): round(float(v), 2) for k, v in pcts.items()}

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
        "total_missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": target_counts,
        "target_percentage": target_pct,
    }
    return summary


if __name__ == "__main__":
    df = load_raw_data()
    print("Dataset Loaded Successfully:")
    print(df.head(5))
    summary = inspect_data(df)
    print("\nDataset Summary:")
    print(f"Total Rows: {summary['rows']}")
    print(f"Total Columns: {summary['columns']}")
    print(f"Target Distribution: {summary['target_distribution']}")
    print(f"Target Percentages: {summary['target_percentage']}")
