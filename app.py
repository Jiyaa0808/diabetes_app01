import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=60, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=850, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47)
age = st.number_input("Age", min_value=21, max_value=81, value=30)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("Diabetic ❌")
    else:
        st.success("Not Diabetic ✅")
