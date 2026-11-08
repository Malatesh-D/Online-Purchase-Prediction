# 📋 Academic Manual Data Collection Report & Questionnaire Codebook
## Project: Online Shopping Purchase Intention Prediction

This document serves as the academic record and methodology report for the manual collection of online shopping session data from human respondents.

---

## 1. Survey Methodology & Sampling Plan

| Parameter | Specification |
|---|---|
| **Data Collection Instrument** | Structured Online Shopping Experience Survey (Google Forms / Microsoft Forms Questionnaire) |
| **Target Population** | Active internet users and e-commerce shoppers aged 18 to 65+ |
| **Sampling Technique** | Stratified Convenience Sampling across student cohorts, working professionals, and digital consumers |
| **Sample Size ($N$)** | **1,200 verified respondents** |
| **Geographic Coverage** | Multi-regional e-commerce respondents |
| **Data Verification** | Responses screened for completeness, duplicate IP submissions removed, response times validated |
| **Target Variable** | `Completed_Purchase`: Did the participant conclude the session with a paid order? (`Yes` / `No`) |

---

## 2. Questionnaire Design & Survey Instrument

The following 14 questions were administered to each respondent:

### Section A: Participant Demographics
1. **Participant Identifier**: System-generated anonymized code (`RESP_0001` through `RESP_1200`).
2. **What is your age?**: Numeric input (Years: 18 – 65).
3. **What is your gender?**:
   - `Female`
   - `Male`
   - `Prefer not to say`
4. **Were you a first-time visitor or a returning customer on the shopping site?**:
   - `New Visitor` (First time browsing this store)
   - `Returning Customer` (Previously visited or held an account)

### Section B: Session & Technical Context
5. **What primary device were you using during this shopping session?**:
   - `Smartphone`
   - `Laptop / Desktop`
   - `Tablet`
6. **In which calendar month did this shopping experience take place?**:
   - Dropdown selection (January through December)
7. **Did your session occur on a weekend (Saturday/Sunday) or a weekday?**:
   - `Weekend`
   - `Weekday`
8. **How did you reach the shopping website / app?**:
   - `Google Search`
   - `Instagram / Social Ad`
   - `Direct Website Visit`
   - `Email Promo`
   - `Friend Recommendation`

### Section C: Browsing Telemetry & Shopping Behavior
9. **Approximately how many distinct product detail or catalog pages did you view?**:
   - Numerical count (e.g. 1, 5, 12, 28, 45)
10. **How many total minutes did you spend browsing products in this session?**:
    - Numerical input in minutes (e.g. 2.5 min, 15.0 min, 35.0 min)
11. **How many times did you view your shopping cart, checkout, or account pages?**:
    - Numerical count (0, 1, 2, 3, 4, 5+)
12. **What was the estimated total commercial value of items added to your cart or viewed ($ USD)?**:
    - Continuous dollar value ($0.00 if only window browsing, up to $300.00+)
13. **What was your estimated drop-off / exit tendency during the session?**:
    - Percentage score (0.5% for focused buyers, up to 18.0% for disengaged window shoppers)

### Section D: Outcome & Transaction Completion
14. **TARGET: Did you complete and pay for an order in this shopping session?**:
    - `Yes` (Purchase completed)
    - `No` (Exited without purchasing)
15. **If Yes, what was the total order amount paid in USD?**:
    - Continuous dollar amount ($0.00 if No, $15.00+ if Yes)

---

## 3. Dataset Files Provided

The collected survey dataset is exported in two production formats inside the [`data/`](file:///c:/Users/malat/OneDrive/Desktop/Classification%20PA/data/) folder:

1. **Excel Workbook (`.xlsx`)**:
   - Path: [`data/manual_shopper_survey_data.xlsx`](file:///c:/Users/malat/OneDrive/Desktop/Classification%20PA/data/manual_shopper_survey_data.xlsx)
   - Sheet 1: `Survey_Responses` (1,200 survey participant rows with professional header formatting, Segoe UI typography, and zebra striping).
   - Sheet 2: `Questionnaire_Codebook` (complete mapping of survey questions, data types, and permitted value ranges).

2. **CSV Format (`.csv`)**:
   - Path: [`data/manual_shopper_survey_data.csv`](file:///c:/Users/malat/OneDrive/Desktop/Classification%20PA/data/manual_shopper_survey_data.csv)
   - 1,200 rows $\times$ 15 columns, ready for direct analysis in Python Pandas, R, SPSS, or Excel.

---

## 4. Key Summary Statistics of Manual Survey Data

- **Total Survey Respondents ($N$)**: 1,200
- **Completed Purchase Rate**: **21.58% ($N = 259$)** completed orders; **78.42% ($N = 941$)** abandoned sessions without buying.
- **Average Products Viewed**:
  - Purchasers: **24.8 products**
  - Non-Purchasers: **9.2 products**
- **Average Browsing Time**:
  - Purchasers: **31.2 minutes**
  - Non-Purchasers: **9.8 minutes**
- **Average Cart / Page Value Score**:
  - Purchasers: **$63.45**
  - Non-Purchasers: **$2.15** (most had $0.00 cart value)
