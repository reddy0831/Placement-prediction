import streamlit as st
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🎓 Placement Prediction App")

# User inputs
gender = st.selectbox("Gender", encoders["gender"].classes_)
ssc_p = st.slider("SSC Percentage", 40.0, 100.0, 75.0)
ssc_b = st.selectbox("SSC Board", encoders["ssc_b"].classes_)
hsc_p = st.slider("HSC Percentage", 40.0, 100.0, 75.0)
hsc_b = st.selectbox("HSC Board", encoders["hsc_b"].classes_)
hsc_s = st.selectbox("HSC Stream", encoders["hsc_s"].classes_)
degree_p = st.slider("Degree Percentage", 40.0, 100.0, 70.0)
degree_t = st.selectbox("Degree Type", encoders["degree_t"].classes_)
workex = st.selectbox("Work Experience", encoders["workex"].classes_)
etest_p = st.slider("E-Test Percentage", 0.0, 100.0, 70.0)
specialisation = st.selectbox("MBA Specialisation", encoders["specialisation"].classes_)
mba_p = st.slider("MBA Percentage", 40.0, 100.0, 70.0)

# Encode inputs
input_data = [
    encoders["gender"].transform([gender])[0],
    ssc_p,
    encoders["ssc_b"].transform([ssc_b])[0],
    hsc_p,
    encoders["hsc_b"].transform([hsc_b])[0],
    encoders["hsc_s"].transform([hsc_s])[0],
    degree_p,
    encoders["degree_t"].transform([degree_t])[0],
    encoders["workex"].transform([workex])[0],
    etest_p,
    encoders["specialisation"].transform([specialisation])[0],
    mba_p
]

# Predict
if st.button("Predict Placement Status"):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    status = encoders["status"].inverse_transform([prediction])[0]
    st.success(f"🎯 Predicted Placement Status: **{status}**")
