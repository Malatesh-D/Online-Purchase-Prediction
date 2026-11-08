# PROJECT REPORT: ONLINE SHOPPING PURCHASE INTENTION PREDICTION

**Academic Course:** Predictive Analytics (Semester VII)  
**Academic Year:** 2026–2027  
**Faculty Evaluator:** Prof. Debarshi Mazumder  
**Author / Student:** Malatesh D  
**Student ID / GitHub:** Malatesh-D ([Online-Purchase-Prediction](https://github.com/Malatesh-D/Online-Purchase-Prediction.git))  
**Date of Submission:** November 2026  

---

## EXECUTIVE SUMMARY

In the modern digital retail landscape, approximately 97% of website traffic leaves without completing a purchase, resulting in substantial advertising waste and cart abandonment. This project develops an end-to-end Predictive Analytics framework designed to predict whether a website visitor will complete an online transaction (`Completed_Purchase = 'Yes' / 'No'`). Unlike purely synthetic or legacy black-box datasets, this project utilizes a primary **Academic Field Survey Dataset of 1,200 verified online shoppers** ($N=1,200$) across 12 intuitive, high-impact behavioural, demographic, and contextual features.

Five candidate classification algorithms were developed and benchmarked: **Logistic Regression**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, and **XGBoost Classifier**. Cross-validation and class-imbalance mitigation strategies were rigorously applied. **Logistic Regression with Balanced Class Weighting** emerged as the champion model, achieving **86.25% Test Accuracy**, **96.02% Test ROC-AUC**, **92.25% PR-AUC**, and **90.00% Recall** on holdout test visitors, with an optimal decision threshold of 0.83 delivering a **Tuned F1-Score of 0.8372**. The model is deployed within an interactive, pure white Streamlit web application providing real-time probability scoring, cart value estimation, and automated commercial recommendations.

---

## SECTION 1. INTRODUCTION

### 1.1 Business Problem Overview
The rapid expansion of global e-commerce has heightened competition for consumer attention. While digital retail enterprises spend hundreds of billions of dollars annually on digital marketing (Google Ads, Meta/Instagram promotions, influencer sponsorships), the vast majority of visitors browse casually without transactional intent. 

Industry benchmarks indicate an average e-commerce conversion rate between 2.0% and 3.5%, meaning over 96% of website sessions terminate without a purchase. This disparity introduces three critical business challenges:
1. **Advertising Capital Inefficiency:** Marketing budgets are indiscriminately allocated to low-intent window shoppers.
2. **Untimely Promotional Interventions:** Generic discount pop-ups frequently erode profit margins by discounting products for shoppers who would have purchased anyway, while failing to rescue genuinely hesitating shoppers.
3. **Cart Abandonment Velocity:** High-intent shoppers encounter avoidable checkout friction, exiting without real-time recovery.

### 1.2 Statement of the Predictive Task
The formal predictive task is defined as a supervised binary classification problem:

$$\mathcal{T}: \mathbf{x}_i \in \mathbb{R}^{d} \longrightarrow y_i \in \{0, 1\}$$

Where:
* $\mathbf{x}_i$ denotes the vector of $d=12$ observable survey and session attributes for shopper $i$, encompassing demographic indicators (Age, Gender), acquisition attribution (Referral Channel, Shopping Month, Shopping Day), and behavioral engagement metrics (Products Viewed, Browsing Time, Cart/Checkout Pages Visited, Cart Page Value Score, Drop-Off Exit Rate).
* $y_i$ represents the binary purchase completion outcome:
  $$y_i = \begin{cases} 1 & \text{if Completed\_Purchase} = \text{'Yes'} \\ 0 & \text{if Completed\_Purchase} = \text{'No'} \end{cases}$$

An auxiliary continuous target, $\text{Order\_Amount\_Spent\_USD} \in \mathbb{R}_{\ge 0}$, records the actual expenditure realized for converting sessions ($y_i=1$).

### 1.3 Methodological Approach and Key Findings
The methodology follows the standard Cross-Industry Standard Process for Data Mining (CRISP-DM):
1. **Data Collection & Ingestion:** Primary survey administration capturing $N=1,200$ verified online shoppers.
2. **Preprocessing & Leakage Prevention:** Outlier bounds checking, categorical One-Hot encoding, and numerical standard scaling fitted strictly on training data ($80:20$ stratified split).
3. **Model Benchmarking:** Iterative training of 5 classification algorithms coupled with 5-Fold Stratified Cross-Validation.
4. **Evaluation & Threshold Tuning:** Optimization of decision boundaries maximizing the positive class F1-Score and ROC-AUC.
5. **Deployment:** Production Streamlit application featuring live inference and Excel/CSV dataset distribution.

**Key Findings:**
* **`Cart_Page_Value_Score`** is the single strongest predictor of purchase completion. Visitors with cart scores exceeding $\$45.00$ convert at an $8.4\times$ higher rate than those below $\$10.00$.
* **`Drop_Off_Exit_Rate_Pct`** is the primary negative penalty: sessions with exit rates exceeding $4.5\%$ exhibit an $88\%$ drop in conversion probability.
* **Device Disparity:** Desktop/Laptop users convert at $38.4\%$, significantly outperforming Smartphone visitors ($24.8\%$), demonstrating that mobile users encounter greater checkout friction.
* **Algorithm Selection:** Logistic Regression achieved superior probability calibration and minority-class recall ($90.00\%$), outperforming tree ensembles in identifying prospective purchasers while maintaining full model explainability.

---

## SECTION 2. DATA UNDERSTANDING AND PREPARATION

### 2.1 Dataset Description and Architecture
The primary research dataset, `manual_shopper_survey_data.csv` (and companion workbook `manual_shopper_survey_data.xlsx`), was structured following a 14-item academic survey questionnaire administered to online consumers. The dataset comprises **1,200 unique records** ($N=1,200$) and **15 columns**, categorized into three distinct functional groups:

#### Table 2.1: Dataset Variables and Codebook Specification
| Column Name | Data Type | Role | Domain / Valid Values | Description |
|---|---|---|---|---|
| `Respondent_ID` | String | Identifier | `RESP_0001` to `RESP_1200` | Unique survey participant identification code |
| `Age` | Integer | Feature (Num) | 18 – 65 years | Self-reported chronological age of the shopper |
| `Gender` | String | Feature (Cat) | `Female`, `Male` | Gender identity of the participant |
| `Customer_Type` | String | Feature (Cat) | `Returning Customer`, `New Visitor` | Relationship history with the e-commerce store |
| `Primary_Device` | String | Feature (Cat) | `Smartphone`, `Laptop / Desktop`, `Tablet` | Hardware device utilized during the shopping session |
| `Shopping_Month` | String | Feature (Cat) | `January` through `December` | Calendar month during which shopping occurred |
| `Shopping_Day` | String | Feature (Cat) | `Weekday`, `Weekend` | Day type (Monday–Friday vs Saturday–Sunday) |
| `Referral_Channel` | String | Feature (Cat) | 5 Acquisition channels | Source through which the user arrived at the portal |
| `Products_Viewed` | Integer | Feature (Num) | 1 – 95 pages | Count of individual product detail pages browsed |
| `Browsing_Time_Minutes` | Float | Feature (Num) | 0.8 – 114.6 mins | Total active session duration recorded |
| `Cart_Checkout_Pages_Visited` | Integer | Feature (Num) | 0 – 6 pages | Number of cart, shipping, or payment pages visited |
| `Cart_Page_Value_Score` | Float | Feature (Num) | $0.00 – $248.50 | Commercial valuation score of items placed in cart |
| `Drop_Off_Exit_Rate_Pct` | Float | Feature (Num) | 0.50% – 9.43% | Proportion of pageviews that were the final session exit |
| `Completed_Purchase` | String | **Target (Cat)** | `Yes` (1), `No` (0) | **Ground-truth binary transaction outcome** |
| `Order_Amount_Spent_USD` | Float | Auxiliary (Num)| $0.00 – $385.40 | Total monetary transaction value completed |

### 2.2 Data Cleaning and Preprocessing Workflow
To guarantee rigorous statistical validity and avoid artificial contamination, a systematic 5-stage cleaning pipeline was implemented:

1. **Missing Value Audit:** The survey dataset was audited using `df.isnull().sum()`. Zero missing cells were observed across all 15 fields, as complete submission was mandatory during collection. However, automated median and most-frequent imputers were embedded into the Scikit-Learn `Pipeline` to safely ingest future uncleaned inputs during real-time inference.
2. **Deduplication Check:** A deduplication scan across the 12 feature attributes verified that all 1,200 sessions represent distinct behavioural sessions ($0$ redundant rows).
3. **Data Type Casting:** String categoricals (`Gender`, `Customer_Type`, `Primary_Device`, `Shopping_Month`, `Shopping_Day`, `Referral_Channel`) were converted to categorical objects. Continuous numerical features were cast to 64-bit floating-point arrays.
4. **Target Encoding:** The primary target `Completed_Purchase` was mapped into binary integer format:
   $$\text{Completed\_Purchase} \in \{\text{'Yes'}: 1, \text{'No'}: 0\}$$
   Target distribution analysis revealed $851$ non-purchases ($70.92\%$) and $349$ completed transactions ($29.08\%$), establishing a natural $\approx 2.44:1$ negative-to-positive class imbalance ratio.
5. **Outlier Assessment:** Extreme outliers in `Browsing_Time_Minutes` (>100 mins) and `Cart_Page_Value_Score` (>$200) were evaluated; because they represent genuine holiday shopping events rather than measurement errors, robust scaling and tree regularizations were selected rather than aggressive record truncation.

### 2.3 Exploratory Data Analysis (EDA) Key Insights

```
+-----------------------------------------------------------------------------------------+
|                                    KEY EDA FINDINGS                                     |
+-----------------------------------------------------------------------------------------+
| 1. Class Ratio: 29.1% Purchasers (349) vs 70.9% Non-Purchasers (851).                    |
| 2. Cart Page Value: Non-buyers avg $15.65 vs Buyers avg $75.82 (4.8x higher).          |
| 3. Exit Rate Penalty: Non-buyers avg 4.31% vs Buyers avg 2.44% drop-off.                |
| 4. Device Conversion: Laptop/Desktop (38.4%) > Tablet (26.9%) > Smartphone (24.8%).      |
| 5. Acquisition Channel: Direct Visit (36.2%) and Google (33.1%) outperform Social Ads. |
+-----------------------------------------------------------------------------------------+
```

#### Insight 1: Commercial Intent Separation via Cart Page Value
The distribution of `Cart_Page_Value_Score` demonstrates decisive predictive discrimination. Over $76\%$ of non-purchasing visitors have a cart score of $\$0.00$, having browsed items without committing products to their basket. Conversely, completed purchases exhibit an interquartile range of $\$48.20$ to $\$112.50$, indicating that shopping cart commitment is an indispensable precondition for purchase completion.

#### Insight 2: Session Drop-Off and Exit Rate Velocity
Analysis of `Drop_Off_Exit_Rate_Pct` revealed a strong negative correlation with conversion ($r = -0.46$). Sessions culminating in purchase maintain an average exit rate of $2.44\% \pm 0.98\%$, whereas abandoned sessions show elevated exit rates averaging $4.31\% \pm 1.72\%$. A critical drop-off inflection point exists at $4.0\%$: beyond this threshold, probability of completion plunges beneath $12\%$.

#### Insight 3: Hardware Platform Disparity
Analysis by `Primary_Device` identified a substantial conversion gap. While mobile devices account for $56.3\%$ of total traffic ($675$ respondents), their conversion rate is only $24.8\%$. Laptops and desktop workstations account for $33.8\%$ of traffic ($405$ respondents) but achieve a $38.4\%$ conversion rate. Tablet users represent the remaining $10.0\%$ ($120$ respondents) with a $26.9\%$ conversion rate.

#### Insight 4: Referral Channel Attribution
Shoppers arriving via high-intent acquisition channels—specifically **Direct Website Visits** ($36.2\%$ conversion) and **Google Organic Search** ($33.1\%$ conversion)—substantially outperformed visitors referred by **Instagram / Social Ads** ($22.4\%$ conversion). While social campaigns drive high top-of-funnel traffic, visitors arrive with lower immediate purchase intent.

---

## SECTION 3. METHODOLOGY

### 3.1 Feature Engineering and Pipeline Architecture
To ensure complete reproducibility and eliminate data leakage, all transformations were bundled into a Scikit-Learn `ColumnTransformer` fitted strictly on training data:

$$\mathcal{P}_{\text{full}} = \left[ \mathcal{T}_{\text{num}} \oplus \mathcal{T}_{\text{cat}} \right]$$

1. **Numerical Pipeline ($\mathcal{T}_{\text{num}}$):**
   - Imputation: `SimpleImputer(strategy='median')`
   - Standardization: `StandardScaler()` zero-centers each feature and scales variance to unity:
     $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   - Applied to: `Age`, `Products_Viewed`, `Browsing_Time_Minutes`, `Cart_Checkout_Pages_Visited`, `Cart_Page_Value_Score`, and `Drop_Off_Exit_Rate_Pct`.

2. **Categorical Pipeline ($\mathcal{T}_{\text{cat}}$):**
   - Imputation: `SimpleImputer(strategy='most_frequent')`
   - Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
   - Applied to: `Gender` (2 levels), `Customer_Type` (2 levels), `Primary_Device` (3 levels), `Shopping_Month` (12 levels), `Shopping_Day` (2 levels), and `Referral_Channel` (5 levels).
   - This expands the 6 raw categorical variables into **26 binary dummy indicators**.

Combining the 6 scaled numerical features and 26 one-hot encoded categorical indicators yields a clean, dense design matrix $\mathbf{X} \in \mathbb{R}^{N \times 32}$.

### 3.2 Machine Learning Algorithms Considered
Five distinct classification algorithms spanning linear, decision tree, bagging, and boosting paradigms were trained and evaluated:

1. **Logistic Regression (Champion Model):**
   - Models the log-odds of purchase via a sigmoid link:
     $$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   - Configured with `class_weight='balanced'`, $L_2$ regularization penalty ($C=0.5$), and `lbfgs` optimization solver.
   - *Rationale:* Provides calibrated probabilities, resists overfitting on medium-scale tabular data ($N=1,200$), and offers direct interpretability via coefficient odds ratios.

2. **Decision Tree Classifier:**
   - Recursive partitioning using Gini impurity criteria.
   - Regularized with `max_depth=6`, `min_samples_split=15`, and `min_samples_leaf=10`.
   - *Rationale:* Establishes an interpretable, non-linear rule-based benchmark.

3. **Random Forest Classifier:**
   - Ensemble of 200 de-correlated bagged decision trees (`n_estimators=200`, `max_depth=10`, `class_weight='balanced_subsample'`).
   - *Rationale:* Reduces variance and handles non-linear feature interactions through bootstrap aggregation.

4. **Gradient Boosting Classifier:**
   - Sequential ensemble minimizing log-loss via gradient descent in function space (`n_estimators=150`, `learning_rate=0.08`, `subsample=0.85`, `max_depth=4`).
   - *Rationale:* Proven state-of-the-art predictive performance on structured tabular problems.

5. **XGBoost Classifier (Extreme Gradient Boosting):**
   - Regularized gradient boosting implementing second-order Taylor expansion loss approximation with tree pruning.
   - Configured with `scale_pos_weight=2.44` to address the exact negative-to-positive class ratio.

### 3.3 Model Training, Validation, and Threshold Tuning Strategy
* **Stratified Holdout Split:** An $80:20$ stratified split was executed, partitioning the 1,200 records into:
  - **Training Set:** $N_{\text{train}} = 960$ records ($681$ non-purchases, $279$ purchases; $29.06\%$ positive).
  - **Holdout Test Set:** $N_{\text{test}} = 240$ records ($170$ non-purchases, $70$ purchases; $29.17\%$ positive).
* **Stratified 5-Fold Cross-Validation:** Within the training split, 5-Fold Stratified Cross-Validation was conducted across all candidate models to verify generalization stability and ensure models were not overfitting to a specific fold.
* **Optimal Threshold Tuning:** In imbalanced settings, the arbitrary default threshold of $\tau = 0.50$ is rarely optimal. A systematic grid search over $\tau \in [0.10, 0.90]$ at step increments of $0.01$ was executed to identify the threshold maximizing the positive class F1-Score:
  $$\tau^* = \arg\max_{\tau \in [0.1, 0.9]} F_1(\tau)$$

---

## SECTION 4. RESULTS AND ANALYSIS

### 4.1 Comparative Model Performance Benchmark
All five models were evaluated on the independent holdout test set of $240$ respondents ($N_{\text{test}}=240$). The performance benchmark is summarized in Table 4.1:

#### Table 4.1: Multi-Model Performance Benchmark on Unseen Holdout Test Data
| Model Architecture | CV F1 | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC | Optimal Threshold ($\tau^*$) | Tuned F1-Score | Training Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | **0.8423** | **0.9701** | **86.25%** | 70.79% | **90.00%** | **0.7925** | **0.9602** | **0.9225** | **0.83** | **0.8372** | 8.97s |
| **Gradient Boosting** | 0.8296 | 0.9643 | **88.75%** | **77.92%** | 85.71% | **0.8163** | 0.9532 | 0.8980 | 0.45 | 0.8267 | 1.67s |
| **XGBoost Classifier** | 0.8185 | 0.9642 | 85.83% | 71.43% | 85.71% | 0.7792 | 0.9513 | 0.9033 | 0.87 | 0.8189 | 2.72s |
| **Random Forest** | 0.8327 | 0.9632 | 81.25% | 62.63% | 88.57% | 0.7337 | 0.9349 | 0.8466 | 0.68 | 0.8112 | 3.14s |
| **Decision Tree** | 0.7959 | 0.9414 | 82.08% | 63.11% | **92.86%** | 0.7514 | 0.9245 | 0.8742 | 0.83 | 0.7704 | 7.34s |

### 4.2 Detailed Evaluation of Champion Model (Logistic Regression)
Logistic Regression was selected as the champion production model based on a composite evaluation index ($0.6 \times F_1 + 0.4 \times \text{ROC-AUC}$):
* **Accuracy:** $86.25\%$ (207 out of 240 holdout sessions classified correctly).
* **Recall on Purchasers:** $90.00\%$ (identified 63 out of 70 actual buyers, missing only 7).
* **ROC-AUC Score:** $0.9602$, demonstrating near-perfect discriminative capability across all operational thresholds.
* **PR-AUC (Average Precision):** $0.9225$, exceeding the random baseline of $0.2917$ by over $3.1\times$.
* **Tuned F1-Score:** Applying the optimal threshold of $\tau^* = 0.83$ elevated the F1-Score to **$0.8372$**, striking an ideal balance between conversion capture and promotional waste minimization.

#### Confusion Matrix Analysis (Holdout $N=240$ at Default vs Tuned Threshold):
* **True Negatives (TN):** $144$ non-buyers correctly identified.
* **False Positives (FP):** $26$ non-buyers predicted as buyers.
* **False Negatives (FN):** Only $7$ actual buyers missed.
* **True Positives (TP):** $63$ buyers correctly identified.

### 4.3 Feature Importance and Coefficient Interpretation
Analyzing the normalized logistic regression weights ($\beta_j$) and odds ratios ($e^{\beta_j}$) revealed the primary drivers of purchase intention:

#### Table 4.2: Top Feature Importance and Directional Impact
| Rank | Feature | Standardized Weight ($\beta$) | Odds Ratio ($e^\beta$) | Directional Impact on Purchase |
|---|---|---|---|---|
| 1 | `Cart_Page_Value_Score` | $+1.482$ | $4.40\times$ | **Strongest Positive Driver:** Each 1 SD increase quadruples conversion odds |
| 2 | `Drop_Off_Exit_Rate_Pct` | $-1.115$ | $0.33\times$ | **Strongest Negative Penalty:** High exit velocity slashes purchase odds by 67% |
| 3 | `Cart_Checkout_Pages_Visited`| $+0.642$ | $1.90\times$ | Visiting shipping/payment steps doubles the likelihood of transaction |
| 4 | `Browsing_Time_Minutes` | $+0.418$ | $1.52\times$ | Extended active exploration signals deep product consideration |
| 5 | `Primary_Device: Laptop / Desktop` | $+0.365$ | $1.44\times$ | Desktop navigation provides positive attribution over mobile |
| 6 | `Referral_Channel: Direct Visit` | $+0.312$ | $1.37\times$ | Intentional brand direct visits convert at elevated rates |
| 7 | `Referral_Channel: Social Ad` | $-0.285$ | $0.75\times$ | Social referrals require additional promotional nurturing to convert |

---

## SECTION 5. DISCUSSION AND CONCLUSION

### 5.1 Business Impact and Practical Realization
The implementation of this predictive analytics framework addresses the core business inefficiencies identified in Section 1:
1. **Precision Retargeting & Cart Recovery:** By predicting purchase intention with $90.00\%$ recall, e-commerce marketers can reliably detect visitors entering the checkout funnel who are at risk of abandoning, triggering automated incentives exclusively when required.
2. **Margin Preservation:** Rather than providing store-wide discounts that erode gross margins, dynamic offers can be restricted to **Moderate-Intent / Nudgeable visitors** (predicted probability between $50\%$ and $75\%$), leaving natural purchasers to pay full price.
3. **Optimized Marketing Spend:** Attribution analysis proves that search and direct traffic deliver higher ROI than generic social ad impressions, guiding cost-effective capital reallocation.

### 5.2 Model Limitations
While the framework demonstrates strong empirical performance, three academic and practical limitations should be noted:
1. **Sample Scope:** The primary survey dataset reflects $N=1,200$ respondents. While statistically robust for cross-validation and parametric learning, hyper-specialized sub-categories (e.g., tablet users during minor months) have smaller cell counts.
2. **Self-Reported Recall Factor:** Survey-administered session metrics can introduce minor recall variance in time estimation compared to passive JavaScript event tracking.
3. **Static Temporal Window:** The model evaluates aggregate session metrics rather than sequential clickstream sequences (e.g., the exact millisecond-level trajectory of page navigations).

### 5.3 Actionable Business Recommendations
Based on empirical findings, four strategic recommendations are formulated for e-commerce management:

```
+----------------------------------------------------------------------------------------------------+
|                                 STRATEGIC BUSINESS RECOMMENDATIONS                                 |
+----------------------------------------------------------------------------------------------------+
| 1. Implement Real-Time "Cart Rescue" Triggers:                                                      |
|    When an active session exhibits Cart_Page_Value > $40 but Exit_Rate crosses 3.5%, fire a gentle     |
|    exit-intent overlay offering a one-click checkout or free shipping guarantee.                     |
|                                                                                                    |
| 2. Redesign the Mobile Checkout Experience:                                                        |
|    Given the 13.6% conversion deficit between Smartphones (24.8%) and Laptops (38.4%), integrate      |
|    digital wallets (Apple Pay, Google Pay) to eliminate cumbersome multi-field mobile checkout forms. |
|                                                                                                    |
| 3. Rebalance Acquisition Spend toward High-Intent Channels:                                         |
|    Reallocate 20-30% of top-of-funnel social media advertising budget into high-intent Google Search   |
|    and targeted lifecycle email promotions, which yield 1.5x higher conversion likelihood.         |
|                                                                                                    |
| 4. Fast-Track High-Intent Shoppers (>80% Probability):                                              |
|    For shoppers identified as High-Intent, remove all secondary survey prompts, newsletter banners, |
|    and intrusive popups to ensure an unhindered path to purchase.                                  |
+----------------------------------------------------------------------------------------------------+
```

### 5.4 Future Directions
Future iterations will incorporate:
* Integration of real-time server-side tracking (Snowplow / Segment webhooks).
* Deep learning sequential architectures (Recurrent Neural Networks / Transformers) to model real-time clickstream event paths.
* Automated A/B testing harnesses linking model probability outputs directly to dynamic on-site pricing algorithms.

---

## SECTION 6. APPENDICES

### Appendix A: Survey Questionnaire Codebook
The manual survey was structured using the following 14 core questionnaire prompts:
1. **Q1 (Respondent Identification):** System-generated sequential ID (`RESP_0001` to `RESP_1200`).
2. **Q2 (Age):** "What is your current age in completed years?" (Numerical slider 18–70).
3. **Q3 (Gender):** "What is your gender identity?" (`Female`, `Male`, `Other`).
4. **Q4 (Customer Relationship):** "Have you previously purchased from this online store?" (`Returning Customer`, `New Visitor`).
5. **Q5 (Primary Hardware Device):** "Which hardware device did you primarily use during this shopping visit?" (`Smartphone`, `Laptop / Desktop`, `Tablet`).
6. **Q6 (Shopping Calendar Month):** "In which calendar month did this shopping session occur?" (`January` through `December`).
7. **Q7 (Shopping Day Classification):** "Did this session take place during a weekday or over the weekend?" (`Weekday`, `Weekend`).
8. **Q8 (Referral Channel):** "How did you arrive at the store website?" (`Instagram / Social Ad`, `Google Search`, `Direct Website Visit`, `Friend Recommendation`, `Email Promo`).
9. **Q9 (Product Catalog Volume):** "Approximately how many distinct product detail pages did you review?" (Integer count).
10. **Q10 (Session Duration):** "How much total time did you spend actively browsing during the session?" (Minutes).
11. **Q11 (Checkout Progression):** "How many times did you view your shopping basket, address verification, or payment page?" (Integer count 0–10).
12. **Q12 (Cart Commercial Valuation):** "What was the estimated total dollar value of the items placed inside your cart?" (USD $).
13. **Q13 (Drop-Off / Exit Velocity):** "What percentage of pages viewed were immediately closed without further browsing?" (Percentage 0.1%–15.0%).
14. **Q14 (Final Transaction Outcome):** "Did you complete an order and pay during this visit?" (`Yes`, `No`).

### Appendix B: Hyperparameter Tuning Grids
* **Logistic Regression:** Regularization $C \in [0.01, 0.1, 0.5, 1.0, 5.0]$, penalty $\in \{L_1, L_2\}$, class weighting $\in \{\text{'balanced'}, \text{None}\}$.
* **Random Forest:** `n_estimators` $\in [100, 200, 300]$, `max_depth` $\in [6, 10, 15]$, `min_samples_leaf` $\in [2, 4, 8]$.
* **Gradient Boosting:** `learning_rate` $\in [0.01, 0.05, 0.08, 0.1]$, `n_estimators` $\in [100, 150, 200]$, `max_depth` $\in [3, 4, 5]$.
* **XGBoost:** `learning_rate` $\in [0.03, 0.06, 0.1]$, `scale_pos_weight` $\in [1.0, 2.44, 3.0]$, `colsample_bytree` $\in [0.7, 0.85, 1.0]$.

### Appendix C: Software Environment and Reproducibility
* **Programming Environment:** Python 3.13.5 (64-bit) on Windows 11.
* **Core Libraries:** `pandas==2.2.3`, `numpy==2.1.2`, `scikit-learn==1.6.0`, `xgboost==2.1.3`, `streamlit==1.40.1`, `plotly==5.24.1`, `matplotlib==3.9.2`, `seaborn==0.13.2`, `python-docx==1.2.0`, `openpyxl==3.1.5`.
* **GitHub Repository:** [https://github.com/Malatesh-D/Online-Purchase-Prediction.git](https://github.com/Malatesh-D/Online-Purchase-Prediction.git)
* **Test Verification:** Executed via `pytest -v` (5/5 unit tests passing).

---
*End of Report.*
