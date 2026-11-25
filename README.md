# ChurnGuard — Customer Churn Prediction

ChurnGuard is a Python and Streamlit machine learning application that estimates whether a telecom customer is likely to leave a service. It uses the IBM Telco Customer Churn dataset and presents a prediction with a churn probability and a simple retention-oriented suggestion.

## Project Overview

Customer churn can affect a telecom company's customer base and revenue. This project explores how supervised machine learning can help identify customers who may be at risk of leaving.

## Features

- Enter customer details through a Streamlit interface.
- Generate a churn prediction.
- View the predicted churn probability.
- Display a simple retention recommendation.

## Machine Learning

The project explores these classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

The existing project documentation identifies **Logistic Regression** as the selected final model based on its overall evaluation. No performance metric is stated here because a verified score is not included in the project summary.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Repository Structure

- `app.py` — Streamlit application.
- `data/` — project dataset resources.
- `experiments/` — model experiment materials.
- `models/` — saved model resources.
- `notebooks/` — notebook-based exploration.
- `inspect_data.py`, `preprocess_data.py`, `prepare_features.py`, `split_data.py` — data preparation workflow.
- `requirements.txt` — Python dependencies.

## Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/ShreyaSingh46/ChurnGuard-Customer-Churn-Prediction.git
   cd ChurnGuard-Customer-Churn-Prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Learning Goals

This project brings together data preprocessing, feature preparation, classification models, model selection, and a simple interactive interface.

> **Note:** ChurnGuard is an educational machine learning project. Predictions are estimates based on the dataset and model, not guarantees about an individual customer's behaviour.
