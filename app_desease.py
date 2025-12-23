# -*- coding: utf-8 -*-
"""
Created on Sunday September 01 15:29:50 2024
@author: Alphatech
"""

import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# =========================
# Load Models Safely
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_model = joblib.load(os.path.join(BASE_DIR, 'diabetes_model_new.pkl'))
heartdisease_model = joblib.load(os.path.join(BASE_DIR, 'Heart_Disease.pkl'))
parkinson_model = joblib.load(os.path.join(BASE_DIR, 'Parkinson_model.pkl'))

# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    selected = option_menu(
        'Machine Learning Techniques',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# =========================
# Diabetes Prediction Page
# =========================
if selected == 'Diabetes Prediction':

    st.markdown(
        "<div style='background-color:purple;padding:10px'>"
        "<h2 style='color:white;text-align:center;'>Diabetes Prediction</h2></div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Pregnancies', 0, 20, 1)
        SkinThickness = st.number_input('Skin Thickness', 0, 100, 20)
        DPF = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5)

    with col2:
        Glucose = st.number_input('Glucose', 0, 200, 100)
        Insulin = st.number_input('Insulin', 0, 846, 79)
        Age = st.number_input('Age', 1, 120, 33)

    with col3:
        BloodPressure = st.number_input('Blood Pressure', 0, 122, 72)
        BMI = st.number_input('BMI', 0.0, 100.0, 32.0)

    if st.button('Predict Diabetes'):
        features = np.array([[Pregnancies, Glucose, BloodPressure,
                              SkinThickness, Insulin, BMI, DPF, Age]])
        prediction = diabetes_model.predict(features)[0]
        prob = diabetes_model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error(f"Diabetes Detected (Probability: {prob:.2f})")
        else:
            st.success(f"No Diabetes (Probability: {1 - prob:.2f})")

# =========================
# Heart Disease Prediction
# =========================
if selected == 'Heart Disease Prediction':

    st.markdown(
        "<div style='background-color:purple;padding:10px'>"
        "<h2 style='color:white;text-align:center;'>Heart Disease Prediction</h2></div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', 1, 100, 40)
        cholesterol = st.number_input('Cholesterol', 100, 700, 200)
        HR = st.number_input('Max Heart Rate', 50, 200, 100)

    with col2:
        ChestPain = st.number_input('Chest Pain Type', 1, 4, 1)
        FBS = st.number_input('FBS > 120', 0, 1, 0)
        Exercise = st.number_input('Exercise Angina', 0, 1, 0)

    with col3:
        BP = st.number_input('Blood Pressure', 80, 200, 120)
        EKG = st.number_input('EKG', 0, 2, 0)
        Depression = st.number_input('ST Depression', 0.0, 6.0, 1.0)

    slope = st.number_input('Slope', 1, 3, 1)
    vessels = st.number_input('Vessels', 0, 3, 0)
    thallium = st.number_input('Thallium', 1, 7, 3)

    if st.button('Predict Heart Disease'):
        features = np.array([[age, ChestPain, BP, cholesterol, FBS,
                              EKG, HR, Exercise, Depression,
                              slope, vessels, thallium]])
        prediction = heartdisease_model.predict(features)[0]
        prob = heartdisease_model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error(f"Heart Disease Detected (Probability: {prob:.2f})")
        else:
            st.success(f"No Heart Disease (Probability: {1 - prob:.2f})")

# =========================
# Parkinson's Prediction
# =========================
if selected == 'Parkinsons Prediction':

    st.markdown(
        "<div style='background-color:brown;padding:10px'>"
        "<h2 style='color:white;text-align:center;'>Parkinson's Prediction</h2></div>",
        unsafe_allow_html=True
    )

    fo = st.number_input('MDVP:Fo(Hz)', 0.0, 500.0, 95.0)
    fhi = st.number_input('MDVP:Fhi(Hz)', 0.0, 500.0, 195.0)
    flo = st.number_input('MDVP:Flo(Hz)', 0.0, 500.0, 75.0)
    jitter = st.number_input('Jitter(%)', 0.0, 0.1, 0.0078)
    shimmer = st.number_input('Shimmer', 0.0, 0.2, 0.03)
    nhr = st.number_input('NHR', 0.0, 1.0, 0.0247)
    hnr = st.number_input('HNR', 0.0, 55.0, 21.33)
    rpde = st.number_input('RPDE', 0.0, 1.0, 0.31)
    dfa = st.number_input('DFA', 0.0, 1.0, 0.37)
    ppe = st.number_input('PPE', 0.0, 1.0, 0.65)

    if st.button('Predict Parkinsons'):
        features = np.array([[fo, fhi, flo, jitter, shimmer,
                              nhr, hnr, rpde, dfa, ppe]])
        prediction = parkinson_model.predict(features)[0]
        prob = parkinson_model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error(f"Parkinson's Detected (Probability: {prob:.2f})")
        else:
            st.success(f"No Parkinson's (Probability: {1 - prob:.2f})")

# =========================
# Footer
# =========================
st.markdown(
    "<div style='background-color:black;padding:10px'>"
    "<h5 style='color:white;text-align:center;'>© 2024 Created by: Odeyale Kehinde Musiliudeen</h5></div>",
    unsafe_allow_html=True
)
