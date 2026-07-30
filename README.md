#  Customer Churn Prediction — Telco Dataset

Predicting which telecom customers are likely to cancel their service, using IBM's public Telco
Customer Churn dataset — an end-to-end classification project covering EDA, class-imbalance
handling, model benchmarking, hyperparameter tuning, and business-driven model selection.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

📓 [View the full notebook](https://github.com/Abdoul-razaq/Customer-Churn-Prediction/blob/main/Customer_Churn_Prediction.ipynb)
|
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144sqx8HxYM3pTmiKlxJbWgVwZpK4EWGz?usp=sharing)
---

##  Project Overview

Each row in this dataset represents a **telecom customer**. Given information about a customer —
their account details (contract type, tenure, billing), the services they subscribe to (internet,
streaming, tech support, etc.), and basic demographics — the goal is to predict whether that
customer will **churn** (cancel their service).

This is a supervised **binary classification** problem: the model predicts a category
(`Churn` / `No Churn`), not a number.

**Dataset:** 7,043 customers × 21 raw features (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), a widely
used churn-modeling benchmark released by IBM.

##  Business Problem

Customer churn is one of the most significant threats to revenue in subscription-based
industries — the telecom sector alone sees an annual churn rate of 15–25%, and acquiring a new
customer typically costs far more than retaining an existing one. This project builds a model
that flags at-risk customers *before* they leave, so a business can prioritize retention efforts
where they matter most.

##  Objectives

- Explore the data to understand what drives churn (contract type, services, tenure, charges)
- Clean and encode the data for modeling
- Benchmark a wide range of classification algorithms
- Address the class imbalance in the churn label (~26.6% churn rate)
- Select a model based on the business-relevant metric — **recall on the churn class**, since
  missing a true churner is costlier than a false alarm
- Tune the selected model and interpret which features drive its predictions
- Report final performance on a held-out test set and translate results into business actions

##  Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core programming language |
| **Google Colab / Jupyter Notebook** | Interactive development environment |
| **pandas** | Data loading & manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Data visualization |
| **scikit-learn** | Preprocessing, models, evaluation |
| **XGBoost / CatBoost** | Gradient-boosted tree models |
| **imbalanced-learn (SMOTE)** | Class-imbalance handling |
| **Optuna** | Bayesian hyperparameter optimization |

##  Methodology

### 1. Exploratory Data Analysis (EDA)
Examined the 7,043 customers across demographics, subscribed services, contract/billing details,
and tenure — confirming that month-to-month contracts, fiber-optic internet, high monthly charges,
and the absence of add-on services (tech support, online security) are strongly associated with
churn. A mixed-type association matrix (Cramér's V / correlation ratio / Spearman) was built from
scratch to quantify relationships between categorical and numerical features.

### 2. Data Cleaning
- Converted `TotalCharges` from text to numeric and dropped the small number of rows with missing
  values (new customers with zero tenure)
- Dropped `customerID` (a unique identifier with no predictive value)
- Consolidated `"No internet service"` / `"No phone service"` into a single `"No"` category

### 3. Feature Engineering
- One-hot encoded all categorical features
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) with `MinMaxScaler`

### 4. Train/Test Split (Stratified)
Split the data **80/20**, stratified on the churn label so both sets preserve the true ~26.6%
churn rate.

### 5. Baseline Model Benchmarking
Benchmarked **13 classification algorithms** — Logistic Regression, Ridge Classifier, Linear SVM,
Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, Extra Trees, Bagging,
CatBoost, Voting, and Stacking — using a shared evaluation pipeline, ranked by recall on the
churn class.

### 6. Class Imbalance Handling (SMOTE)
Applied **SMOTE** to the training set only (never the test set) to balance the ~26.6% churn rate,
then re-benchmarked all 13 models to measure the recall gain.

### 7. Feature Importance
Inspected both tree-based feature importances and Logistic Regression coefficients — confirming
that **contract length, tenure, fiber-optic internet, monthly charges, and lack of tech
support/online security** are the strongest churn drivers, consistent with the EDA.

### 8. Hyperparameter Tuning
Tuned the selected model (**AdaBoost**, trained on SMOTE-balanced data) with **Optuna** using
5-fold stratified cross-validation, optimizing for recall.

### 9. Final Held-Out Evaluation
Evaluated the tuned model **once** on the untouched test set and saved the final model + scaler
for reuse.

##  Results

**Selected model:** AdaBoost (trained on SMOTE-balanced data), tuned with Optuna
(`n_estimators=338`, `learning_rate≈0.0152`, best CV recall: 0.873)

**Final held-out test performance:**

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.91 | 0.46 |
| Recall | 0.66 | **0.82** |
| F1-score | 0.76 | 0.59 |

- **Overall accuracy:** 70%
- The model correctly flags **82% of actual churners** — the priority metric for a retention
  use case — at the cost of some false positives, a deliberate trade-off given that missing a
  real churner is costlier than an unnecessary retention offer.

##  Business Recommendations

- **Target month-to-month customers early** — this segment churns far more than annual/2-year
  contract holders; consider incentivizing longer commitments
- **Bundle protective services** (tech support, online security, device protection) — customers
  without these churn more
- **Watch fiber-optic customers with high monthly charges ($70–110)** — an elevated-risk segment
- **Prioritize senior citizens** in retention campaigns — they churn at nearly double the rate of
  non-seniors
- **Operationalize the model** as a monthly scoring system, routing the highest predicted-risk
  customers to a retention team

##  Repository Structure

```
├── Customer_Churn_Prediction.ipynb   # Main notebook (EDA → modeling → evaluation)
├── data/                              # Dataset (or download instructions, if not committed)
├── README.md
└── requirements.txt
```

##  Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Customer_Churn_Prediction.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
xgboost
catboost
imbalanced-learn
optuna
```

##  Key Takeaways

The specific dataset (2020-era Telco churn snapshot) is really just the practice ground — the
more important outcome of this project is the **disciplined process**:
- Choosing a business-relevant metric (recall on churn) instead of defaulting to accuracy on an
  imbalanced target
- Handling class imbalance correctly — applying SMOTE only to training data, never the test set
- Comparing many algorithms systematically rather than assuming one is best
- Reporting a single, honest final evaluation on a held-out test set

