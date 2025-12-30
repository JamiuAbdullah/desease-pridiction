# -*- coding: utf-8 -*-


import joblib
import streamlit as st
from streamlit_option_menu import option_menu  # type: ignore
import numpy as np

# Load models
diabetes_model = joblib.load('diabetes_model_new.pkl')
heartdisease_model = joblib.load('Heart_Disease.pkl')
parkinson_model = joblib.load('Parkinson_model.pkl')

def footer(text):
    st.markdown(f"""
    <div style="background-color:black; padding:10px; margin-top:30px">
        <h5 style="color:white; text-align:center;">
            {text}
        </h5>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Machine Learning Techniques',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

if selected == 'Diabetes Prediction':
    st.markdown("""
        <div style="background-color:purple; padding:10px">
        <h2 style="color:white; text-align:center;">Machine Learning Model to predict Diabetes</h2>
        </div>
    """, unsafe_allow_html=True)

    st.write("Enter the values below to predict the likelihood of diabetes:")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', 0, 20, 1)
        SkinThickness = st.number_input('Skin Thickness (mm)', 0, 100, 20)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    with col2:
        Glucose = st.number_input('Glucose Level', 0, 200, 100)
        Insulin = st.number_input('Insulin Level (mu U/ml)', 0, 846, 79)
        Age = st.number_input('Age', 0, 120, 33)
    with col3:
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', 0, 122, 72)
        BMI = st.number_input('Body Mass Index (BMI)', 0.0, 100.0, 32.0)

    if st.button('Predict Diabetes'):
        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = diabetes_model.predict(features)[0]

        probability = None
        if hasattr(diabetes_model, "predict_proba"):
            probability = diabetes_model.predict_proba(features)[0][1]

        if prediction == 1:
            if probability is not None:
                st.success(f"Diabetes detected (probability: {probability:.2f})")
            else:
                st.success("Diabetes detected")
        else:
            if probability is not None:
                st.success(f"No diabetes detected (probability: {1 - probability:.2f})")
            else:
                st.success("No diabetes detected")

    footer("© 2025 Diabetes Prediction System | Created by Odeyale Kehinde Musiliudeen")                

if selected == 'Heart Disease Prediction':
    st.markdown("""
        <div style="background-color:purple; padding:10px">
        <h2 style="color:white; text-align:center;">Heart Disease Prediction using Machine Learning</h2>
        </div>
    """, unsafe_allow_html=True)

    st.write("Enter the values below to predict likelihood of heart disease:")

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input('Age', 0, 120, 40)
        cholesterol = st.number_input('Cholesterol level', 100, 700, 100)
        slope = st.number_input('Slope of ST', 1, 3, 1)
    with col2:
        ChestPain = st.number_input('Chest pain type', 1, 4, 1)
        FBS = st.number_input('FBS over 120', 0, 1, 0)
        vessels = st.number_input('Number of vessels', 0, 3, 0)
    with col3:
        BP = st.number_input('BP level', 100, 500, 100)
        EKG = st.number_input('EKG result', 0, 2, 0)
        thallium = st.number_input('Thallium', 1, 10, 1)
    with col1:
        HR = st.number_input('Max HR', 50, 200, 100)
    with col2:
        Exercise = st.number_input('Exercise angina', 0, 1, 0)
    with col3:
        Depression = st.number_input('ST depression', 0.0, 2.0, 1.0)

    if st.button('Predict Heart Disease'):
        features = np.array([[Age, ChestPain, BP, cholesterol, FBS, EKG, HR, Exercise, Depression, slope, vessels, thallium]])
        prediction = heartdisease_model.predict(features)[0]

        probability = None
        if hasattr(heartdisease_model, "predict_proba"):
            probability = heartdisease_model.predict_proba(features)[0][1]

        if prediction == 1:
            if probability is not None:
                st.success(f"Heart disease detected (probability: {probability:.2f})")
            else:
                st.success("Heart disease detected")
        else:
            if probability is not None:
                st.success(f"No heart disease detected (probability: {1 - probability:.2f})")
            else:
                st.success("No heart disease detected")

    footer("© 2025 Heart Disease Prediction System | Created by Jamiu Abdullah Baba")            

if selected == 'Parkinsons Prediction':
    st.markdown("""
        <div style="background-color:brown; padding:10px">
        <h2 style="color:white; text-align:center;">Parkinson's Disease Prediction using ML</h2>
        </div>
    """, unsafe_allow_html=True)

    st.write("Enter the values below to predict the likelihood of Parkinson's:")

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_fo = st.number_input('MDVP:Fo(Hz)', 0.0, 500.0, 95.0)
        MDVP_jitter = st.number_input('MDVP:Jitter(%)', 0.001, 0.00100, 0.001)
        MDVP_PPQ = st.number_input('MDVP:PPQ', 0.001, 0.01000, 0.0071)
        MDVP_Shimmer_db = st.number_input('MDVP:Shimmer(dB)', 0.01, 1.0, 0.246)
        MDVP_Apq = st.number_input('MDVP:APQ', 0.001, 0.1000, 0.0359)
        HNR = st.number_input('HNR', 0.01, 55.0, 21.33)
        D2 = st.number_input('D2', 0.001, 0.900000, 0.0375)
    with col2:
        MDVP_fhi = st.number_input('MDVP:Fhi(Hz)', 0.0, 500.0, 95.0)
        MDVP_jitter_Abs = st.number_input('MDVP:Jitter(Abs)', 0.00001, 0.00090, 0.00002)
        Jitter_DDP = st.number_input('Jitter:DDP', 0.001, 0.01000, 0.007)
        Shimmer_Apq3 = st.number_input('Shimmer:APQ3', 0.001, 0.1000, 0.001)
        Shimmer_DDA = st.number_input('Shimmer:DDA', 0.001, 0.1000, 0.039)
        RPDE = st.number_input('RPDE', 0.0001, 0.900000, 0.03173)
        PPE = st.number_input('PPE', 0.001, 0.900000, 0.6593)
    with col3:
        MDVP_flo = st.number_input('MDVP:Flo(Hz)', 0.0, 500.0, 74.0)
        MDVP_RAP = st.number_input('MDVP:RAP', 0.001, 0.01000, 0.0037)
        MDVP_Shimmer = st.number_input('MDVP:Shimmer', 0.001, 0.01000, 0.007)
        Shimmer_Apq5 = st.number_input('Shimmer:APQ5', 0.001, 0.10000, 0.01)
        NHR = st.number_input('NHR', 0.001, 0.1000, 0.027)
        DFA = st.number_input('DFA', 0.0001, 0.900000, 0.0374)

    if st.button('Predict Parkinson'):
        features = np.array([[
            MDVP_fo, MDVP_fhi, MDVP_flo,
            MDVP_jitter, MDVP_jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
            MDVP_Shimmer, MDVP_Shimmer_db,
            Shimmer_Apq3, Shimmer_Apq5, MDVP_Apq, Shimmer_DDA,
            NHR, HNR, RPDE, DFA, D2, PPE
        ]], dtype=np.float64)

        prediction = parkinson_model.predict(features)[0]

        probability = None
        if hasattr(parkinson_model, "predict_proba"):
            probability = parkinson_model.predict_proba(features)[0][1]

        if prediction == 1:
            if probability is not None:
                st.success(f"Parkinson detected (probability: {probability:.2f})")
            else:
                st.success("Parkinson detected")
        else:
            if probability is not None:
                st.success(f"No Parkinson detected (probability: {1 - probability:.2f})")
            else:
                st.success("No Parkinson detected")

    footer("© 2025 Parkinson’s Prediction System | Created by Jamiu Abdullah Baba")            



