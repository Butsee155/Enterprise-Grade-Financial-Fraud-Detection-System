import streamlit as st
import joblib
import numpy as np

st.title("💳 Fraud Detection System")

model = joblib.load("models/LightGBM.pkl")

st.write("Enter transaction values (30 features)")

input_data = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    prediction = model.predict([input_data])
    prob = model.predict_proba([input_data])[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ Fraud detected! Risk score: {prob:.4f}")
    else:
        st.success(f"Safe transaction. Risk score: {prob:.4f}")