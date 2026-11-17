# RUNTIME TERMINAL EXECUTION OUTPUTS

**Project Title:** Predict Whether a Person Will Make an Online Purchase Using Machine Learning  
**Dataset:** Academic Field Survey Dataset ($N=1,200$)  
**Student:** Malatesh D  
**Faculty Evaluator:** Prof. Debarshi Mazumder  
**Course:** Predictive Analytics (Academic Year 2026–2027)  

---

## 1. RUNTIME OUTPUT: DATA INGESTION & DATASET INSPECTION

**Command Executed:**
```bash
python src/data_loader.py
```

**Terminal Output:**
```text
[2026-11-17 23:06:47] INFO - online_purchase_prediction: Loaded survey dataset with shape (1200, 15) from C:\Users\malat\OneDrive\Desktop\Classification PA\data\manual_shopper_survey_data.csv
Dataset Loaded Successfully:
  Respondent_ID  Age  ... Completed_Purchase Order_Amount_Spent_USD
0     RESP_0001   23  ...                 No                    0.0
1     RESP_0002   41  ...                 No                    0.0
2     RESP_0003   28  ...                 No                    0.0
3     RESP_0004   26  ...                 No                    0.0
4     RESP_0005   21  ...                 No                    0.0

[5 rows x 15 columns]

Dataset Summary:
Total Rows: 1200
Total Columns: 15
Target Distribution: {'No': 851, 'Yes': 349}
Target Percentages: {'No': 70.92, 'Yes': 29.08}
Total Missing Values: 0
Duplicates Found: 0
```

---

## 2. RUNTIME OUTPUT: DATA PREPROCESSING & LEAKAGE-FREE PIPELINE

**Command Executed:**
```bash
python src/preprocessing.py
```

**Terminal Output:**
```text
[2026-11-17 23:06:55] INFO - online_purchase_prediction: Loaded survey dataset with shape (1200, 15) from C:\Users\malat\OneDrive\Desktop\Classification PA\data\manual_shopper_survey_data.csv
[2026-11-17 23:06:55] INFO - online_purchase_prediction: Data split: Train=960 samples (29.06% positive), Test=240 samples (29.17% positive)
[2026-11-17 23:06:55] INFO - online_purchase_prediction: Fitted ColumnTransformer preprocessor on X_train (data leakage prevented).
[2026-11-17 23:06:55] INFO - online_purchase_prediction: Artifact successfully saved to C:\Users\malat\OneDrive\Desktop\Classification PA\models\preprocessor.pkl
Preprocessing verification:
X_train transformed shape: (960, 32)
Transformed features count: 32
Transformed feature names: [
    'num__Age', 'num__Products_Viewed', 'num__Browsing_Time_Minutes', 
    'num__Cart_Checkout_Pages_Visited', 'num__Cart_Page_Value_Score', 'num__Drop_Off_Exit_Rate_Pct', 
    'cat__Gender_Female', 'cat__Gender_Male', 
    'cat__Customer_Type_New Visitor', 'cat__Customer_Type_Returning Customer', 
    'cat__Primary_Device_Laptop / Desktop', 'cat__Primary_Device_Smartphone', 'cat__Primary_Device_Tablet', 
    'cat__Shopping_Month_April', 'cat__Shopping_Month_August', 'cat__Shopping_Month_December', 
    'cat__Shopping_Month_February', 'cat__Shopping_Month_January', 'cat__Shopping_Month_July', 
    'cat__Shopping_Month_June', 'cat__Shopping_Month_March', 'cat__Shopping_Month_May', 
    'cat__Shopping_Month_November', 'cat__Shopping_Month_October', 'cat__Shopping_Month_September', 
    'cat__Shopping_Day_Weekday', 'cat__Shopping_Day_Weekend', 
    'cat__Referral_Channel_Direct Website Visit', 'cat__Referral_Channel_Email Promo', 
    'cat__Referral_Channel_Friend Recommendation', 'cat__Referral_Channel_Google Search', 
    'cat__Referral_Channel_Instagram / Social Ad'
]
```

---

## 3. RUNTIME OUTPUT: MODEL TRAINING, CROSS-VALIDATION & BENCHMARKING

**Command Executed:**
```bash
python src/train.py
```

**Terminal Output:**
```text
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Starting complete model training pipeline...
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Loaded survey dataset with shape (1200, 15) from C:\Users\malat\OneDrive\Desktop\Classification PA\data\manual_shopper_survey_data.csv
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Data split: Train=960 samples (29.06% positive), Test=240 samples (29.17% positive)
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Fitted ColumnTransformer preprocessor on X_train (data leakage prevented).
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Artifact successfully saved to C:\Users\malat\OneDrive\Desktop\Classification PA\models\preprocessor.pkl
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Class ratio in training data: 681 non-purchase to 279 purchase (Weight: 2.44)
[2026-11-17 23:08:32] INFO - online_purchase_prediction: Training and validating Logistic Regression...
[2026-11-17 23:08:40] INFO - online_purchase_prediction: Logistic Regression -> Test Acc: 86.25%, Rec: 90.00%, F1: 0.7925, ROC-AUC: 0.9602, Tuned F1: 0.8372
[2026-11-17 23:08:40] INFO - online_purchase_prediction: Training and validating Decision Tree...
[2026-11-17 23:08:45] INFO - online_purchase_prediction: Decision Tree -> Test Acc: 82.08%, Rec: 92.86%, F1: 0.7514, ROC-AUC: 0.9245, Tuned F1: 0.7704
[2026-11-17 23:08:45] INFO - online_purchase_prediction: Training and validating Random Forest...
[2026-11-17 23:08:50] INFO - online_purchase_prediction: Random Forest -> Test Acc: 81.25%, Rec: 88.57%, F1: 0.7337, ROC-AUC: 0.9349, Tuned F1: 0.8112
[2026-11-17 23:08:50] INFO - online_purchase_prediction: Training and validating Gradient Boosting...
[2026-11-17 23:08:53] INFO - online_purchase_prediction: Gradient Boosting -> Test Acc: 88.75%, Rec: 85.71%, F1: 0.8163, ROC-AUC: 0.9532, Tuned F1: 0.8267
[2026-11-17 23:08:53] INFO - online_purchase_prediction: Training and validating XGBoost...
[2026-11-17 23:08:56] INFO - online_purchase_prediction: XGBoost -> Test Acc: 85.83%, Rec: 85.71%, F1: 0.7792, ROC-AUC: 0.9513, Tuned F1: 0.8189
[2026-11-17 23:08:56] INFO - online_purchase_prediction: Model comparison saved to C:\Users\malat\OneDrive\Desktop\Classification PA\reports\model_comparison.csv
[2026-11-17 23:08:56] INFO - online_purchase_prediction: ==> Selected Best Model: Logistic Regression (Composite F1/ROC-AUC)
[2026-11-17 23:08:56] INFO - online_purchase_prediction: Artifact successfully saved to C:\Users\malat\OneDrive\Desktop\Classification PA\models\best_model.pkl
[2026-11-17 23:08:56] INFO - online_purchase_prediction: JSON data successfully saved to C:\Users\malat\OneDrive\Desktop\Classification PA\models\model_metadata.json
[2026-11-17 23:08:56] INFO - online_purchase_prediction: Generating evaluation charts...
[2026-11-17 23:08:57] INFO - online_purchase_prediction: Saved ROC curves to C:\Users\malat\OneDrive\Desktop\Classification PA\reports\visualizations\roc_curves.png
[2026-11-17 23:08:57] INFO - online_purchase_prediction: Saved PR curves to C:\Users\malat\OneDrive\Desktop\Classification PA\reports\visualizations\pr_curves.png
[2026-11-17 23:09:00] INFO - online_purchase_prediction: Saved confusion matrices grid to C:\Users\malat\OneDrive\Desktop\Classification PA\reports\visualizations\confusion_matrices.png
[2026-11-17 23:09:00] INFO - online_purchase_prediction: Saved feature importance plot to C:\Users\malat\OneDrive\Desktop\Classification PA\reports\visualizations\feature_importance.png
[2026-11-17 23:09:00] INFO - online_purchase_prediction: Saved evaluation text report to C:\Users\malat\OneDrive\Desktop\Classification PA\reports\evaluation_report.txt
[2026-11-17 23:09:00] INFO - online_purchase_prediction: Training and evaluation pipeline completed successfully!

--- MODEL PERFORMANCE COMPARISON ---
              Model  CV F1-Score  CV ROC-AUC  Test Accuracy  Test Precision  Test Recall  Test F1-Score  Test ROC-AUC  Test PR-AUC  Optimal Threshold  Tuned F1-Score  Training Time (s)
Logistic Regression       0.8423      0.9701         0.8625          0.7079       0.9000         0.7925        0.9602       0.9225               0.83          0.8372               7.09
  Gradient Boosting       0.8296      0.9643         0.8875          0.7792       0.8571         0.8163        0.9532       0.8980               0.45          0.8267               1.94
            XGBoost       0.8185      0.9642         0.8583          0.7143       0.8571         0.7792        0.9513       0.9033               0.87          0.8189               3.04
      Random Forest       0.8327      0.9632         0.8125          0.6263       0.8857         0.7337        0.9349       0.8466               0.68          0.8112               4.37
      Decision Tree       0.7959      0.9414         0.8208          0.6311       0.9286         0.7514        0.9245       0.8742               0.83          0.7704               5.29

Best Model: Logistic Regression with metadata saved.
```

---

## 4. RUNTIME OUTPUT: REAL-TIME INFERENCE ENGINE ON SHOPPER PROFILES

**Command Executed:**
```bash
python src/predict.py
```

**Terminal Output:**
```text
[2026-11-17 23:07:43] INFO - online_purchase_prediction: Artifact successfully loaded from C:\Users\malat\OneDrive\Desktop\Classification PA\models\best_model.pkl
[2026-11-17 23:07:43] INFO - online_purchase_prediction: Artifact successfully loaded from C:\Users\malat\OneDrive\Desktop\Classification PA\models\preprocessor.pkl
[2026-11-17 23:07:43] INFO - online_purchase_prediction: PurchasePredictor initialized with model 'LogisticRegression' and threshold 0.83

--- Testing High Intent Buyer Sample ---
predicted_class: Likely to Purchase
completed_purchase: Yes
raw_class: 1
purchase_probability: 0.9756
non_purchase_probability: 0.0244
purchase_probability_pct: 97.6%
non_purchase_probability_pct: 2.4%
confidence_score: 97.6
confidence_score_pct: 97.6%
threshold_used: 0.83
estimated_order_amount: 115.88
estimated_order_amount_usd: $115.88
risk_level: High Intent Shopper
interpretation: Respondent (23 yrs, browsing on Laptop / Desktop via Direct Website Visit) shows very strong commercial commitment (97.6% purchase probability). The high cart value score ($80.70) and multiple checkout steps (2) indicate ready-to-buy intent. Recommendation: Provide 1-click checkout and minimal form friction.
```

---

## 5. RUNTIME OUTPUT: AUTOMATED UNIT TEST SUITE (PYTEST)

**Command Executed:**
```bash
pytest -v
```

**Terminal Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-8.4.2, pluggy-1.5.0 -- C:\Users\malat\miniconda33\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\malat\OneDrive\Desktop\Classification PA
plugins: anyio-4.12.1, asyncio-1.4.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 5 items

tests/test_prediction.py::test_predictor_initialization PASSED           [ 20%]
tests/test_prediction.py::test_predict_structure_and_probabilities PASSED [ 40%]
tests/test_prediction.py::test_missing_field_raises_error PASSED         [ 60%]
tests/test_preprocessing.py::test_clean_dataset PASSED                   [ 80%]
tests/test_preprocessing.py::test_prepare_data_split PASSED              [100%]

============================== 5 passed in 2.77s ==============================
```

---

## 6. RUNTIME OUTPUT: COMPLETE VISUALIZATION & CHART GENERATION SUITE

**Command Executed:**
```bash
python generate_charts.py
```

**Terminal Output:**
```text
================================================================================
📊 STARTING COMPREHENSIVE PROJECT VISUALIZATION GENERATION
================================================================================
Loaded survey dataset: 1200 rows, 15 columns.

Generating [01. Target Class Distribution]... ✅ Saved (129.6 KB)
Generating [02. Shopper Demographics Analysis]... ✅ Saved (195.0 KB)
Generating [03. Primary Device Conversion Rates]... ✅ Saved (152.6 KB)
Generating [04. Marketing Referral Channel Performance]... ✅ Saved (132.1 KB)
Generating [05. Monthly & Weekend Seasonality]... ✅ Saved (296.6 KB)
Generating [06. Cart Page Value Score Distribution]... ✅ Saved (272.1 KB)
Generating [07. Drop-Off Exit Rate Analysis]... ✅ Saved (173.8 KB)
Generating [08. Cart Value vs. Exit Rate Scatter]... ✅ Saved (625.4 KB)
Generating [09. Browsing Time & Products Viewed]... ✅ Saved (180.5 KB)
Generating [10. Checkout Funnel Progression]... ✅ Saved (122.7 KB)
Generating [11. Pearson Correlation Matrix]... ✅ Saved (211.6 KB)
Generating [12. ML Model Benchmark Comparison]... ✅ Saved (145.0 KB)
Generating [13. ROC & Precision-Recall Curves]... ✅ Saved (433.5 KB)
Generating [14. Confusion Matrices Grid]... ✅ Saved (234.7 KB)
Generating [15. Feature Importance & Logistic Weights]... ✅ Saved (268.3 KB)
Generating [16. Order Amount Spent Distribution]... ✅ Saved (460.5 KB)
Generating [17. Executive 4-in-1 Dashboard]... ✅ Saved (822.4 KB)

================================================================================
🎉 VISUALIZATION PIPELINE COMPLETED IN 16.25 SECONDS
📁 Destination Directory: C:\Users\malat\OneDrive\Desktop\Classification PA\reports\visualizations
📊 Total High-Resolution Figures Generated: 17
================================================================================

Generated Charts Manifest:
   1. 01_target_distribution.png               ( 129.6 KB)
   2. 02_demographics_analysis.png             ( 195.0 KB)
   3. 03_device_conversion.png                 ( 152.6 KB)
   4. 04_referral_channels.png                 ( 132.1 KB)
   5. 05_monthly_seasonality.png               ( 296.6 KB)
   6. 06_cart_value_analysis.png               ( 272.1 KB)
   7. 07_drop_off_exit_rate.png                ( 173.8 KB)
   8. 08_cart_value_vs_exit_rate_scatter.png   ( 625.4 KB)
   9. 09_browsing_engagement.png               ( 180.5 KB)
  10. 10_checkout_funnel_progression.png       ( 122.7 KB)
  11. 11_correlation_matrix.png                ( 211.6 KB)
  12. 12_model_comparison_benchmark.png        ( 145.0 KB)
  13. 13_roc_and_pr_curves.png                 ( 433.5 KB)
  14. 14_confusion_matrices.png                ( 234.7 KB)
  15. 15_feature_importance.png                ( 268.3 KB)
  16. 16_order_amount_distribution.png         ( 460.5 KB)
  17. 00_executive_dashboard_summary.png       ( 822.4 KB)
```

---

## 7. RUNTIME OUTPUT: STREAMLIT WEB APPLICATION SERVER

**Command Executed:**
```bash
python -m streamlit run app.py --server.port 8501
```

**Terminal Output:**
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.165:8501
  External URL: http://43.247.158.70:8501

[2026-11-17 23:29:24] INFO - online_purchase_prediction: Artifact successfully loaded from C:\Users\malat\OneDrive\Desktop\Classification PA\models\best_model.pkl
[2026-11-17 23:29:24] INFO - online_purchase_prediction: Artifact successfully loaded from C:\Users\malat\OneDrive\Desktop\Classification PA\models\preprocessor.pkl
[2026-11-17 23:29:24] INFO - online_purchase_prediction: PurchasePredictor initialized with model 'LogisticRegression' and threshold 0.83
HTTP Request: GET / 200 OK
```

---
*Generated directly from verified terminal executions on Python 3.13.5 (64-bit) under Windows 11.*
