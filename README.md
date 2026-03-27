# 📉 Customer Churn Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

An end-to-end machine learning pipeline that predicts customer churn with **91.4% accuracy** using gradient boosting. Includes exploratory data analysis, feature engineering, model selection, and a Streamlit dashboard for business teams.

---

## 🎯 The Problem

Customer acquisition costs 5–25× more than retaining an existing customer. This model identifies customers at high risk of churning *before* they leave, giving retention teams a 30-day window to act.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 80.2% | 0.71 | 0.68 | 0.69 | 0.84 |
| Random Forest | 88.7% | 0.83 | 0.79 | 0.81 | 0.92 |
| **XGBoost (final)** | **91.4%** | **0.89** | **0.87** | **0.88** | **0.96** |

**Top predictive features:**
1. `contract_type` — month-to-month customers churn 3× more
2. `tenure_months` — churn risk drops sharply after 12 months
3. `monthly_charges` — high charges + low tenure = high risk
4. `num_support_tickets` — strong leading indicator of dissatisfaction

---

## 🚀 Quick Start

```bash
git clone https://github.com/TheCromazone/churn-predictor.git
cd churn-predictor
pip install -r requirements.txt

# Train the model
python train.py

# Launch the prediction dashboard
streamlit run dashboard.py
```

---

## 🏗️ Project Structure

```
churn-predictor/
├── notebooks/
│   └── 01_EDA_and_Feature_Engineering.ipynb  # Full analysis walkthrough
├── src/
│   ├── features.py       # Feature engineering pipeline
│   ├── train.py          # Model training + cross-validation
│   ├── evaluate.py       # Metrics, plots (ROC, confusion matrix)
│   └── predict.py        # Inference on new customer data
├── dashboard.py          # Streamlit business dashboard
├── data/
│   └── telco_churn.csv   # Telco customer churn dataset (IBM)
├── models/
│   └── xgb_churn.pkl     # Saved trained model
├── requirements.txt
└── README.md
```

---

## 🔍 Methodology

1. **EDA** — Missing value analysis, class imbalance (26% churn rate), distribution plots
2. **Feature Engineering** — 14 engineered features including charge ratios, tenure buckets, and interaction terms
3. **Preprocessing** — SMOTE oversampling for class imbalance, StandardScaler for numerics, OrdinalEncoder for categoricals
4. **Model Selection** — 5-fold stratified CV across Logistic Regression, Random Forest, XGBoost, and LightGBM
5. **Hyperparameter Tuning** — Optuna Bayesian optimization (100 trials)
6. **Evaluation** — ROC curve, precision-recall curve, confusion matrix, SHAP feature importance

---

## 📡 Predict on New Data

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor.load("models/xgb_churn.pkl")

customer = {
    "tenure_months": 3,
    "contract_type": "Month-to-month",
    "monthly_charges": 89.50,
    "num_support_tickets": 4,
    "internet_service": "Fiber optic",
}

result = predictor.predict(customer)
print(result)
# {"churn_probability": 0.847, "risk_tier": "HIGH", "top_factors": [...]}
```

---

## 📦 Dataset

Uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features). Included in `data/`.

---

## 🛠️ Tech Stack

- `scikit-learn` — preprocessing, metrics, cross-validation
- `xgboost` — gradient boosting classifier
- `optuna` — hyperparameter optimization
- `shap` — model explainability
- `imbalanced-learn` — SMOTE oversampling
- `streamlit` — business dashboard
- `plotly` — interactive charts

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
