"""
Utility functions for logging, artifact persistence, and helper formatting.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict
import joblib

# Setup standard logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("online_purchase_prediction")


def save_artifact(obj: Any, file_path: Path) -> None:
    """Save an object to disk using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info(f"Artifact successfully saved to {file_path}")


def load_artifact(file_path: Path) -> Any:
    """Load an object from disk using joblib with validation."""
    if not file_path.exists():
        raise FileNotFoundError(f"Requested artifact does not exist at {file_path}")
    obj = joblib.load(file_path)
    logger.info(f"Artifact successfully loaded from {file_path}")
    return obj


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON data successfully saved to {file_path}")


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file does not exist at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_percentage(val: float, decimals: int = 2) -> str:
    """Format float into clean percentage string."""
    return f"{val * 100:.{decimals}f}%"
