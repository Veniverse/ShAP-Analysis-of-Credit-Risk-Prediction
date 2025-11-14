# ShAP-Analysis-of-Credit-Risk-Prediction
Predict high-risk credit accounts using XGBoost and provide interpretable insights with SHAP, offering both accurate predictions and actionable explanations for financial risk management.
Credit Risk Prediction with Interpretable Machine Learning (SHAP Analysis)
Project Overview

This project focuses on predicting high-risk credit accounts using machine learning while providing deep interpretability of model predictions through SHAP (SHapley Additive exPlanations).
The main goal is not only to achieve strong predictive performance but also to make individual loan decisions explainable, which is critical in regulated financial industries.

Problem Statement

In the banking and financial sector, accurately identifying high-risk borrowers is crucial for reducing defaults and managing risk. Traditional black-box models often fail to provide clear explanations for why a particular loan was approved or rejected.

This project addresses that challenge by combining XGBoost, a high-performing classification model, with SHAP to explain both global feature importance and individual prediction rationale.

Dataset

The dataset used is synthetic but modeled on real-world credit data.

Key features include:

loan_amount – Amount requested by borrower

annual_income – Borrower’s yearly income

credit_history_length – Years of credit history

num_open_credit – Number of open credit lines

debt_to_income_ratio – Ratio of debt to income

employment_length – Years of employment

purpose – Purpose of the loan

Target variable: 0 = low-risk, 1 = high-risk

Dataset is included in the repository as credit_risk_synthetic.csv.

Project Approach

Data Preprocessing

Handle missing values

Encode categorical variables

Address class imbalance using SMOTE

Model Training

Trained a XGBoost classifier for high performance

Tuned hyperparameters for optimal AUC, Precision, and Recall

Model Evaluation

Metrics used:

AUC (Area Under Curve) – measures model discrimination

Precision – measures correct high-risk predictions

Recall – captures high-risk cases detected

Interpretability with SHAP

Global feature importance: identifies top features driving risk predictions

Individual prediction explanations: uses waterfall plots for:

High-risk case

Low-risk case

Borderline case

Dependence plots: show feature interactions and nonlinear effects

Insights for Risk Management

Features like loan_amount, debt_to_income_ratio, and credit_history_length have strong predictive influence

Individual explanations highlight why specific loan applications are accepted or rejected

Actionable guidance for adjusting approval thresholds and monitoring borderline cases

Deliverables

Python code for preprocessing, model training, and SHAP analysis (credit_risk_shap_project.py)

Synthetic dataset (credit_risk_synthetic.csv)

Visualizations: SHAP summary plots, waterfall plots, and dependence plots

Report / insights: top features and interpretation for individual cases

Technologies & Libraries Used

Python (pandas, numpy, matplotlib)

Machine Learning: XGBoost

Imbalanced Data Handling: SMOTE (imblearn)

Model Interpretability: SHAP

Project Outcome

Achieved interpretable predictions for high-risk and low-risk loans

Identified the top global features influencing credit risk

Provided clear, actionable insights for the risk management team

Demonstrates industry-standard interpretable ML practices in finance

How to Run

Clone the repository:

git clone https://github.com/yourusername/credit-risk-shap.git


Install required libraries:

pip install pandas numpy matplotlib xgboost shap imbalanced-learn


Run the Python script:

python credit_risk_shap_project.py


Explore SHAP plots and insights in the output.
