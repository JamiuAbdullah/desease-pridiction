# -*- coding: utf-8 -*-
import os
import joblib
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# =========================
# SAFE MODEL LOADING
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {filename}")
        st.stop()
    return joblib.load(path)

diabetes_model = load_model("diabetes_model_new.pkl")
heart_model = load_model("Heart_Disease.pkl")
parkinson_model = load_model("Parkinson_model.pkl")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    selected = option_menu(
        "Machine Learning Techniques",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["activity", "heart", "person"],
        default_index=0
    )

# =========================
# DIABETES
# =========================
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    with col2:
        Glucose = st.number_input("Glucose", 0, 200, 100)
        Insulin = st.number_input("Insulin", 0, 900, 79)
        Age = st.number_input("Age", 1, 120, 33)

    with col3:
        BloodPressure = st.number_input("Blood Pressure", 0, 122, 72)
        BMI = st.number_input("BMI", 0.0, 70.0, 32.0)

    if st.button("Predict Diabetes"):
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin, BMI, DPF, Age]])
        result = diabetes_model.predict(features)[0]
        st.success("Diabetic" if result == 1 else "Not Diabetic")

# =========================
# HEART DISEASE
# =========================
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 40)
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)
        hr = st.number_input("Max HR", 50, 220, 150)

    with col2:
        chest_pain = st.number_input("Chest Pain Type", 1, 4, 1)
        fbs = st.number_input("FBS > 120", 0, 1, 0)
        exercise = st.number_input("Exercise Angina", 0, 1, 0)

    with col3:
        bp = st.number_input("Resting BP", 80, 200, 120)
        ekg = st.number_input("EKG", 0, 2, 0)
        depression = st.number_input("ST Depression", 0.0, 6.0, 1.0)

    slope = st.number_input("Slope", 1, 3, 1)
    vessels = st.number_input("Vessels", 0, 3, 0)
    thallium = st.number_input("Thallium", 1, 7, 3)

    if st.button("Predict Heart Disease"):
        features = np.array([[age, chest_pain, bp, cholesterol, fbs, ekg,
                              hr, exercise, depression, slope, vessels, thallium]])
        result = heart_model.predict(features)[0]
        st.success("Heart Disease Detected" if result == 1 else "No Heart Disease")

# =========================
# PARKINSON'S
# =========================
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 0.0, 300.0, 120.0)
        jitter = st.number_input("Jitter (%)", 0.0, 1.0, 0.005)
        shimmer = st.number_input("Shimmer", 0.0, 1.0, 0.03)

    with col2:
        fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 400.0, 200.0)
        rap = st.number_input("RAP", 0.0, 1.0, 0.002)
        nhr = st.number_input("NHR", 0.0, 1.0, 0.02)

    with col3:
        flo = st.number_input("MDVP:Flo(Hz)", 0.0, 300.0, 80.0)
        hnr = st.number_input("HNR", 0.0, 40.0, 20.0)
        ppe = st.number_input("PPE", 0.0, 1.0, 0.2)

    if st.button("Predict Parkinson's"):
        features = np.array([[fo, fhi, flo, jitter, rap, shimmer, nhr, hnr, ppe]])
        result = parkinson_model.predict(features)[0]
        st.success("Parkinson's Detected" if result == 1 else "No Parkinson's")

st.markdown("---")
st.markdown("© 2024 Created by Odeyale Kehinde Musiliudeen")
