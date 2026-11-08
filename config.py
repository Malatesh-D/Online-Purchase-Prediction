"""
Configuration module for the Online Purchase Intention Prediction project.
Based on the Human-Survey Shopping Dataset (manual collection format).
Defines paths, survey features, hyperparameters, and project constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"

# Ensure all primary directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, VISUALIZATIONS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATA_RAW_FILENAME = "manual_shopper_survey_data.csv"
DATA_FILE_PATH = DATA_DIR / DATA_RAW_FILENAME
DATA_EXCEL_FILENAME = "manual_shopper_survey_data.xlsx"
DATA_EXCEL_PATH = DATA_DIR / DATA_EXCEL_FILENAME

# Legacy UCI dataset reference (kept for benchmark comparison)
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
UCI_DATA_FILE_PATH = DATA_DIR / "online_shoppers_intention.csv"

# Model artifacts paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.txt"

# Target & Identifier Columns
TARGET_COLUMN = "Completed_Purchase"
ID_COLUMN = "Respondent_ID"
AUXILIARY_COLUMNS = ["Order_Amount_Spent_USD"]

# -------------------------------------------------------------
# HUMAN-SURVEY FEATURES (12 INTUITIVE & REALISTIC FEATURES)
# -------------------------------------------------------------
# Numerical Features (6 human-survey metrics):
# 1. Age: Shopper demographic age (18 - 70)
# 2. Products_Viewed: Count of individual items viewed
# 3. Browsing_Time_Minutes: Self-reported or tracked session duration in minutes
# 4. Cart_Checkout_Pages_Visited: Number of cart/checkout verification pageviews
# 5. Cart_Page_Value_Score: Estimated monetary/commercial intent score ($0 - $250)
# 6. Drop_Off_Exit_Rate_Pct: Exit rate percentage (drop-off likelihood 0.1% - 15.0%)
NUMERICAL_FEATURES = [
    "Age",
    "Products_Viewed",
    "Browsing_Time_Minutes",
    "Cart_Checkout_Pages_Visited",
    "Cart_Page_Value_Score",
    "Drop_Off_Exit_Rate_Pct",
]

# Categorical Features (6 intuitive human context & demographic dimensions):
# 1. Gender: Male, Female, Other
# 2. Customer_Type: Returning Customer vs New Visitor
# 3. Primary_Device: Smartphone, Laptop / Desktop, Tablet
# 4. Shopping_Month: Calendar month of the shopping visit
# 5. Shopping_Day: Weekday vs Weekend
# 6. Referral_Channel: How the respondent arrived at the store
CATEGORICAL_FEATURES = [
    "Gender",
    "Customer_Type",
    "Primary_Device",
    "Shopping_Month",
    "Shopping_Day",
    "Referral_Channel",
]

# Combined 12 feature columns
ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Friendly labels and descriptions for UI / reporting
FEATURE_DESCRIPTIONS = {
    "Age": "Shopper age in years (Demographic)",
    "Gender": "Gender identity of the shopper (Male / Female / Other)",
    "Customer_Type": "Returning customer or first-time visitor",
    "Primary_Device": "Primary shopping device used (Smartphone / Laptop / Tablet)",
    "Shopping_Month": "Month of shopping activity",
    "Shopping_Day": "Timing of session (Weekday vs Weekend)",
    "Referral_Channel": "Acquisition channel (Social Ad, Google, Direct, Referral, Email)",
    "Products_Viewed": "Total number of product catalog pages viewed during the session",
    "Browsing_Time_Minutes": "Total duration spent shopping in minutes",
    "Cart_Checkout_Pages_Visited": "Number of cart, billing, or checkout verification steps visited",
    "Cart_Page_Value_Score": "Estimated commercial intent score based on cart contents ($)",
    "Drop_Off_Exit_Rate_Pct": "Percentage rate at which visitor leaves pages without interaction (%)",
}

# Categorical Options for UI Dropdowns
GENDER_OPTIONS = ["Female", "Male", "Other"]
CUSTOMER_TYPE_OPTIONS = ["Returning Customer", "New Visitor"]
DEVICE_OPTIONS = ["Smartphone", "Laptop / Desktop", "Tablet"]
MONTH_OPTIONS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
DAY_OPTIONS = ["Weekday", "Weekend"]
REFERRAL_OPTIONS = [
    "Instagram / Social Ad",
    "Google Search",
    "Friend Recommendation",
    "Direct Website Visit",
    "Email Promo",
]

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
