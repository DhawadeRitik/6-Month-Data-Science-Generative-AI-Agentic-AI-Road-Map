import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load trained model
model = joblib.load(r"D:\NIT Course\Projects\Customer Churn\XGBoost_final.pkl")

# Add background image using CSS
# st.markdown("""
#     <style>
#     .stApp {
#         background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }

#     /* Style input elements */
#     .stRadio > label, .stSelectbox > label, .stNumberInput > label {
#         color: black !important;
#         font-weight: bold;
#     }

#     h1 {
#         color: black;
#     }

#     </style>
# """, unsafe_allow_html=True)



# Input Preprocessing

def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    
    df['gender']           = df['gender'].map({'Female': 0, 'Male': 1})
    df['SeniorCitizen']    = df['SeniorCitizen'].map({'No': 0, 'Yes': 1})
    df['Partner']          = df['Partner'].map({'No': 0, 'Yes': 1})
    df['Dependents']       = df['Dependents'].map({'No': 0, 'Yes': 1})
    df['PhoneService']     = df['PhoneService'].map({'No': 0, 'Yes': 1})
    df['MultipleLines']    = df['MultipleLines'].map({'No': 0, 'Yes': 2, 'No phone service': 1})
    df['OnlineSecurity']   = df['OnlineSecurity'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    df['OnlineBackup']     = df['OnlineBackup'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    df['DeviceProtection'] = df['DeviceProtection'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    df['TechSupport']      = df['TechSupport'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    df['StreamingTV']      = df['StreamingTV'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    df['StreamingMovies']  = df['StreamingMovies'].map({'No': 0, 'Yes': 2, 'No internet service': 1})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    df['InternetService']  = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract']         = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod']    = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })

    return df

# Title of App
st.title("📊 Customer Churn Prediction")
# st.markdown(
#     "<h1 style='color:white;'>📊 Customer Churn Prediction</h1>",
#     unsafe_allow_html=True
# )


# Input Form

st.subheader("Please fill out the customer details:")

with st.expander("🧑‍💼 Customer Info"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender         = st.radio("Gender", ['Male', 'Female'])
        senior_citizen = st.radio("Senior Citizen", ['No', 'Yes'])
        partner        = st.radio("Partner", ['No', 'Yes'])

    with col2:
        dependents     = st.radio("Dependents", ['No', 'Yes'])
        phone_service  = st.radio("Phone Service", ['No', 'Yes'])
        multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'Yes', 'No'])

    with col3:
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        contract         = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        payment_method   = st.selectbox("Payment Method", [
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ])

with st.expander("🌐 Internet & Streaming Services"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tech_support      = st.radio("Tech Support", ['No', 'Yes', 'No internet service'])
        online_security   = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])

    with col2:
        online_backup     = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])

    with col3:
        streaming_tv      = st.radio("Streaming TV", ['No', 'Yes', 'No internet service'])
        streaming_movies  = st.radio("Streaming Movies", ['No', 'Yes', 'No internet service'])

with st.expander("💰 Billing Info"):
    col1, col2 = st.columns(2)

    with col1:
        paperless_billing = st.radio("Paperless Billing", ['Yes', 'No'])
        monthly_charges   = st.number_input("Monthly Charges", min_value=0.0)

    with col2:
        total_charges     = st.number_input("Total Charges", min_value=0.0)
        tenure_range      = st.number_input("Tenure Range", min_value=0)

# Prediction Button

if st.button("🔮 Predict"):
    # Collect input data
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_range': tenure_range
    }

    # Preprocess and predict
    processed_data = preprocess_input(user_data)
    prediction = model.predict(processed_data)

    # Display results
    if prediction[0] == 1:
        st.markdown(
            "<div style='background-color:#ffcccc;padding:15px;border-radius:10px'><h4 style='color:red;'>🚨 The customer is likely to churn.</h4></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#ccffcc;padding:15px;border-radius:10px'><h4 style='color:green;'>✅ The customer is likely to stay.</h4></div>",
            unsafe_allow_html=True
        )
