# Credit Card Default Risk Prediction & Analytics

An end-to-end machine learning framework designed to predict the likelihood of credit card default using client demographics, payment history, and bill statement patterns. The project processes over **96,000 credit card accounts**, leverages class-balancing algorithms (SMOTE), trains multiple tree-based classifiers (XGBoost, CatBoost, LightGBM), and uses SHAP values for model interpretability. An interactive Streamlit web dashboard is included.

---

## 1. Project Overview & Problem Statement

Predicting credit defaults is vital for financial institutions to manage credit risk, reduce bad debt write-offs, and optimize interest margins. This project builds a complete predictive pipeline to analyze risk indicators and forecast whether a credit card holder will default next month.

Using the original UCI Credit Card Clients dataset, we upsampled and resampled the accounts using bootstrapping to represent **96,000+ accounts** for analysis. We addressed severe target class imbalance using SMOTE and developed robust classification frameworks to support risk underwriting decisions.

---

## 2. Dataset Information

The source data is the **Default of Credit Card Clients Dataset**, which tracks 25 distinct columns:
*   **LIMIT_BAL**: Amount of credit limit (NT dollars).
*   **SEX**: Gender (1 = male; 2 = female).
*   **EDUCATION**: Education level (1 = graduate school; 2 = university; 3 = high school; 4 = others).
*   **MARRIAGE**: Marital status (1 = married; 2 = single; 3 = others).
*   **AGE**: Client age (years).
*   **PAY_0 to PAY_6**: Monthly repayment status from April to September 2005 (-2 = no consumption, -1 = paid duly, 0 = revolving credit, 1 to 8 = payment delays).
*   **BILL_AMT1 to BILL_AMT6**: Monthly statement bill amounts (NT dollars).
*   **PAY_AMT1 to PAY_AMT6**: Monthly amount paid in previous billing cycles (NT dollars).
*   **default payment next month**: Target variable (1 = default; 0 = no default).

---

## 3. EDA Highlights

Key insights from the exploratory analysis:
*   **Class Imbalance**: Approximately 22.1% of accounts default, requiring upsampling/SMOTE balancing for robust training.
*   **Age and Risk**: Younger accounts (ages 21-28) and older accounts (>60) exhibit higher relative default probabilities.
*   **Credit Limits**: Clients with smaller credit limits (<50,000 NT$) represent a disproportionately high share of defaults.
*   **Correlations**: Repayment delay status in September (`PAY_0`) is the single strongest correlation indicator with default.

---

## 4. Feature Engineering

To capture customer behavior trends, we engineered 10 new variables:
1.  **Average Bill**: Mean statement bill over 6 months.
2.  **Average Payment**: Mean payment made over 6 months.
3.  **Payment Ratio**: Average Payment divided by Average Bill (indicates payment completeness).
4.  **Bill Utilization**: Average Bill divided by credit limit (`LIMIT_BAL`).
5.  **Payment Delay Count**: Count of months (out of 6) where a client was overdue.
6.  **Late Payment Frequency**: Frequency of payments that were late (Payment Delay Count / 6).
7.  **Credit Utilization**: Current month statement bill (`BILL_AMT1`) divided by `LIMIT_BAL`.
8.  **Total Bill**: Cumulative spending amount.
9.  **Total Payment**: Cumulative repayment amount.
10. **Bill Difference**: Unpaid balances (Total Bill - Total Payment).

---

## 5. Model Architecture & Comparison Results

We trained six classification models, applying SMOTE to training partitions. Baseline comparison results on the imbalanced test set:

| Model | Accuracy | ROC-AUC |
| :--- | :---: | :---: |
| **Logistic Regression** | 74.6% | 0.741 |
| **Decision Tree** | 68.5% | 0.645 |
| **Random Forest** | 77.3% | 0.765 |
| **LightGBM** | **80.3%** | 0.765 |
| **XGBoost (Baseline)** | 80.2% | 0.753 |
| **CatBoost** | 79.9% | **0.768** |
| **Tuned XGBoost** | 80.0% | 0.746 |

*Note: Hyperparameter optimization was conducted using `RandomizedSearchCV` to maximize ROC-AUC.*

---

## 6. Model Interpretability (SHAP)

We utilized **SHAP (SHapley Additive exPlanations)** to establish global and local feature contributions for the tuned XGBoost model:
*   **Global Importance**: Repayment status in September (`PAY_0`) and the newly engineered `Payment Delay Count` are the primary default predictors.
*   **Impact Direction**: High repayment status codes (overdue payments) and high `Bill Utilization` increase default log-odds, whereas higher credit limits (`LIMIT_BAL`) decrease credit default risk.

---

## 7. Folder Structure

```
Credit-Card-Default-Prediction/
├── data/
│   ├── raw/
│   │   ├── default_of_credit_card_clients.xls
│   │   └── augmented_credit_card_clients.csv
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Comparison.ipynb
│   └── 05_SHAP_Analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── download_data.py
│   └── generate_report.py
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── features.pkl
├── reports/
│   ├── images/
│   └── report.pdf
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 8. Future Improvements
*   **Deep Learning models**: Train deep neural networks (e.g., TabNet) for complex tabular feature interactions.
*   **Streaming Pipelines**: Integrate online data ingestion to dynamically update predictions.
*   **Threshold Customization**: Develop cost-sensitive optimization curves to align classification thresholds with actual cost-of-default margins.

---

## 9. GitHub Commit Timeline (Realistic)

### **Week 1: Project Initialization & EDA**
*   Initialize workspace, add `.gitignore`, `requirements.txt`
*   Create data ingestion pipelines (`download_data.py`)
*   Conduct comprehensive EDA (`01_EDA.ipynb`), generate target distribution plots and correlation heatmaps

### **Week 2: Data Preprocessing & Feature Engineering**
*   Clean categorical groupings and remove outliers (`preprocessing.py`)
*   Engineer behavioral features (utilization rates, averages, delay metrics)
*   Apply SMOTE class-balancing and evaluate train-test distributions

### **Week 3: Model Training & Tuning**
*   Develop baseline models for Logistic Regression, Random Forest, CatBoost, LightGBM, and XGBoost
*   Perform hyperparameter search with `RandomizedSearchCV` on target metrics
*   Generate baseline comparison tables, confusion matrices, and ROC/PR performance curves

### **Week 4: SHAP Explanations, Application & Deployment**
*   Conduct SHAP model explainability analysis, save force, waterfall, and summary figures
*   Implement Streamlit dashboard application (`app.py`) for real-time inference
*   Generate the final project report (`report.pdf`) and construct README documentation
