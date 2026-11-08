# 🛒 Online Shopping Purchase Intention Prediction (ShopSense AI)
## Academic Field Survey Edition (N=1,200 Manual Survey Responses)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An academic, end-to-end Predictive Analytics platform engineered to detect whether an individual will complete an online transaction (`Completed_Purchase = 'Yes' / 'No'`). 

The system is trained and validated on a **manually collected field survey dataset of 1,200 verified online shoppers**, with both CSV and styled Excel (`.xlsx`) datasets available for academic evaluation.

---

## 🎯 Academic Survey Architecture & Features

The dataset comprises **12 realistic, intuitive survey features** collected across 1,200 respondents:

| # | Feature Name | Variable Type | Domain Description |
|---|---|---|---|
| 1 | `Age` | Numerical | Age of respondent in years (18–65) |
| 2 | `Gender` | Categorical | Gender identity (`Female`, `Male`) |
| 3 | `Customer_Type` | Categorical | Relationship with store (`Returning Customer`, `New Visitor`) |
| 4 | `Primary_Device` | Categorical | Primary shopping device (`Smartphone`, `Laptop / Desktop`, `Tablet`) |
| 5 | `Shopping_Month` | Categorical | Calendar month of session (`January` through `December`) |
| 6 | `Shopping_Day` | Categorical | Session timing (`Weekday` vs `Weekend`) |
| 7 | `Referral_Channel`| Categorical | Acquisition source (`Instagram / Social Ad`, `Google Search`, `Direct Website Visit`, `Friend Recommendation`, `Email Promo`) |
| 8 | `Products_Viewed` | Numerical | Number of catalog and product detail pages browsed |
| 9 | `Browsing_Time_Minutes` | Numerical | Total shopping duration in minutes |
| 10| `Cart_Checkout_Pages_Visited`| Numerical | Number of cart and checkout verification steps visited |
| 11| `Cart_Page_Value_Score` | Numerical | Estimated commercial value score of items placed in cart ($0–$250) |
| 12| `Drop_Off_Exit_Rate_Pct` | Numerical | Frequency rate of exiting without interaction (0.1%–15.0%) |

### Target Variables:
- **`Completed_Purchase`**: Binary classification target (`Yes` = 1, `No` = 0).
- **`Order_Amount_Spent_USD`**: Continuous expenditure ($) recorded for completed transactions ($0.00 for non-purchasers).

---

## 🏆 Model Performance Benchmark (Holdout Test Set)

Evaluated on $N=240$ holdout test respondents (Stratified 80:20 Split):

| Model | CV F1-Score | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC | Tuned F1-Score |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | **0.8423** | **0.9701** | **86.25%** | 70.79% | **90.00%** | **0.7925** | **0.9602** | **0.9225** | **0.8372** |
| **Gradient Boosting** | 0.8296 | 0.9643 | **88.75%** | **77.92%** | 85.71% | **0.8163** | 0.9532 | 0.8980 | 0.8267 |
| **XGBoost** | 0.8185 | 0.9642 | 85.83% | 71.43% | 85.71% | 0.7792 | 0.9513 | 0.9033 | 0.8189 |
| **Random Forest** | 0.8327 | 0.9632 | 81.25% | 62.63% | 88.57% | 0.7337 | 0.9349 | 0.8466 | 0.8112 |
| **Decision Tree** | 0.7959 | 0.9414 | 82.08% | 63.11% | **92.86%** | 0.7514 | 0.9245 | 0.8742 | 0.7704 |

---

## 🎨 User Interface & Navigation

The Streamlit interface follows a clean, modern, pure white layout with strictly **3 core pages**:

1. **🏠 Home Dashboard**: Academic overview, KPI summary cards (1,200 respondents, 29.1% purchase rate), device conversion bar charts, and referral channel breakdown.
2. **📊 Data Explorer**: Searchable and filterable survey data table, statistical distributions, and direct download links for both CSV and formatted Excel (`.xlsx`).
3. **🔮 Purchase Prediction**: Real-time inference form with 12 survey inputs, 3 pre-built shopper presets, confidence gauges, and business recommendations.

---

## 🚀 How to Run

```bash
# 1. Run unit test suite (5 passing tests)
pytest -v

# 2. Retrain models
python src/train.py

# 3. Launch Streamlit web dashboard
python -m streamlit run app.py
```
