
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Page settings
st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)

# Header
st.title("🛡️ ChurnGuard")
st.subheader("Customer Intelligence & Retention Platform")

st.write(
    "Predict whether a customer is likely to leave "
    "your telecom service."
)

st.divider()

st.header("Customer Churn Prediction")

st.write("Enter the customer's details below.")

# Customer input form
with st.form("customer_form"):

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )

    with col2:
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)", 0, 72, 12
        )

        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    st.divider()
    st.subheader("Service Information")

    col1, col2 = st.columns(2)

    with col1:
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()
    st.subheader("Billing Information")

    col1, col2 = st.columns(2)

    with col1:
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

    with col2:
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

# Prediction
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()
    st.header("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error(
            "⚠️ This customer is likely to leave."
        )
        st.write(
            "Recommendation: Consider offering "
            "a suitable retention offer or loyalty benefit."
        )
    else:
        st.success(
            "✅ This customer is likely to stay."
        )
        st.write(
            "Recommendation: Continue providing "
            "good customer service."
        )

    st.progress(float(probability))

    st.caption(
        "Prediction generated using the trained "
        "Logistic Regression model."
    )
