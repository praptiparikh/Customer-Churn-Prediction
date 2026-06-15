# Customer Churn Prediction Analysis
Predicted customer churn for 7,043 telecom customers using Logistic Regression and Random Forest (80% accuracy, 0.84 ROC-AUC); identified key churn drivers through EDA and feature importance analysis.

## Overview
This project analyzes customer churn behavior for a telecom company using the Telco Customer Churn dataset. It covers data cleaning, feature engineering, exploratory data analysis, and builds two classification models to predict which customers are likely to churn.

## Business Questions Answered
- What percentage of customers are churning?
- Does tenure (how long a customer stays) affect churn likelihood?
- Do customers with higher monthly charges churn more?
- Which features are the strongest predictors of churn?
- Which model — Logistic Regression or Random Forest — performs better?

## Dataset
- Source  : Telco Customer Churn Dataset (Kaggle)
- Records : 7,043 customers
- Churn Rate: 26.5% (1,869 churned customers)

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest)
- Jupyter Notebook

## Process
1. Cleaned TotalCharges column and handled missing values
2. Removed non-predictive customerID column
3. Created tenure_group feature (0-12, 13-24, 25-48, 49-60, 60+ months)
4. Replaced "No internet service" labels with "No" for consistency
5. Encoded binary columns with Label Encoding, multi-category columns with One-Hot Encoding
6. Split data 80/20 with stratified sampling

## Results
|        Model        | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression |   80.1%  |  0.843  |
| Random Forest       |   78.9%  |  0.826  |

## Key Findings
- Customers with shorter tenure churn significantly more
- Higher monthly charges correlate with higher churn
- Logistic Regression slightly outperformed Random Forest on this dataset
- Top predictive features identified via Random Forest feature importance analysis
