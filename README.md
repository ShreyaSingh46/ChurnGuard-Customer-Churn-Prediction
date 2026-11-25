
# ChurnGuard – Customer Intelligence & Retention Platform

## Project Overview
ChurnGuard is a machine learning-based application that predicts whether a telecom customer is likely to leave the service.

## Problem Statement
Customer churn can affect the growth and revenue of telecom companies. This project uses machine learning to identify customers who may leave so that suitable retention steps can be taken.

## Dataset
IBM Telco Customer Churn Dataset (7,043 customer records).

## Algorithms Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Application Features
- Enter customer details.
- Predict whether a customer is likely to churn.
- View churn probability.
- Display a simple retention recommendation.

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Run the application:

   streamlit run app.py

## Final Model
Logistic Regression was selected as the final model based on its overall evaluation performance.
