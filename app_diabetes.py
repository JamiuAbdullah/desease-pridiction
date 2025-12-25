# -*- coding: utf-8 -*-
"""
Created on Sunday September 01 15:29:50 2024

@author: Alphatech
"""

import joblib
import streamlit as st
from streamlit_option_menu import option_menu # type: ignore
import numpy as np
#loading the saved models
#1 - diabetetic model
diabetes_model = joblib.load('diabetes_model_new.pkl')
 


#sidebar for navigation
with st.sidebar:
    selected = option_menu('Machine Learning Techniques',['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],
                           icons = ['activity','heart','person'],default_index=0)
    
#diabetes Prediction page
if(selected == 'Diabetes Prediction'):
    html_temp = """
    <div style="background-color:purple; padding:10px">
    <h2 style="color:white; text-align:center;">Machine Learning Model to predict Diabetes </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
     
    st.write("Enter the values below to predict the likelihood of diabetes:")
    
    #taking input from  user
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
        
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=100)
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=72)
    
    with col1:
       SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
   
    with col2:
       Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, value=79)
   
    with col3:
       BMI = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0, value=32.0)
   
    with col1:
       DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
   
    with col2:
       Age = st.number_input('Age', min_value=0, max_value=120, value=33)
       
       
    # code for prediction
    # Predict button
if st.button('Predict'):
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = diabetes_model.predict(features)
    
    # Check if the model has 'predict_proba' method
    if hasattr(diabetes_model, "predict_proba"):
        probability = diabetes_model.predict_proba(features)[0][1]
    else:
        probability = None

    # Display result
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


     
 
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    heartdisease_model = joblib.load('Heart_Disease.pkl')
    
    # page title
    
    html_temp = """
    <div style="background-color:purple; padding:10px">
    <h2 style="color:white; text-align:center;">Heart Disease Prediction using Machine Learning </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("Enter the value below to pridict likelihood of heart Disease ") 

    col1,col2,col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age', min_value=0, max_value=1000, value=40)
    with col2:
        ChestPain = st.number_input ('Chest pain type ', min_value=1, max_value=4, value=1)
    with col3:
        BP = st.number_input('BP level', min_value=100, max_value=500,value=100)
    with col1:
        cholesterol = st.number_input('cholesterol level', min_value=100, max_value=700, value=100)
    with col2:
        FBS = st.number_input('FBS over 120', min_value=0, max_value=1, value= 0)
    with col3:
        EKG = st.number_input('EKG result', min_value=0, max_value=2, value=0)
    with col1:
        HR = st.number_input('max HR', min_value=50, max_value=200, value=100)
    with col2:
        Exercise = st.number_input('exercise angina', min_value=0, max_value=1, value=0) 
    with col3:
        Depression = st.number_input('ST depression ', min_value=0.0, max_value=2.0, value=1.0)
    with col1:
        slope = st.number_input('slope of ST', min_value=1, max_value=3, value=1)
    with col2:
        vessels = st.number_input('number of vessels furo', min_value=0, max_value=3, value=0)
    with col3:
        thallium = st.number_input('thallum', min_value=1, max_value=10, value=1)


if st.button('Predict', key='predict_button_1'):
    features = np.array([[Age, ChestPain, BP, cholesterol, FBS, EKG, HR, Exercise, Depression, slope, vessels, thallium]])
    prediction = heartdisease_model.predict(features)
    
    # Check if the model has the 'predict_proba' method
    if hasattr(heartdisease_model, "predict_proba"):
        probability = heartdisease_model.predict_proba(features)[0][1]
    else:
        probability = None
    
    # Display result
    if prediction == 1:
        if probability is not None:
            st.markdown(f'<h4 style="color:blue; background-color:#000; font-size:20px;">The model predicts you <strong>have heart disease</strong> with a probability of {probability:.2f}.</h4>', unsafe_allow_html=True)
        else:
            st.markdown('<h4 style="color:blue; background-color:#000; font-size:20px;">The model predicts you <strong>have heart disease</strong>.</h4>', unsafe_allow_html=True)
        st.status('Model prediction completed')
    else:
        if probability is not None:
            st.markdown(f'<h4 style="color:red; background-color:#000; font-size:20px;">The model predicts you <strong>do not have heart disease</strong> with a probability of {1 - probability:.2f}.</h4>', unsafe_allow_html=True)
        else:
            st.markdown('<h4 style="color:red; background-color:#000; font-size:20px;">The model predicts you <strong>do not have heart disease</strong>.</h4>', unsafe_allow_html=True)
        st.status('Model prediction completed')

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):

    parkinson_model= joblib.load('Parkinson_model.pkl')

    
    # page title
   
    html_temp = """
    <div style="background-color:brown; padding:10px">
    <h2 style="color:white; text-align:center;">Parkinson's Disease Prediction using ML </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("Enter the values below to predict the likelihood of parkinson:")

    col1,col2,col3 = st.columns(3)

    with col1:
        MDVP = st.number_input('MDVP:Fo(Hz)', min_value=0.00, max_value=500.000, value=95.0)
    with col2:
        MDVP = st.number_input('MDVP:Fhi(Hz)', min_value=0.00, max_value=500.000, value=95.0)
    with col3:
       MDVP = st.number_input('MDVP:Flo(Hz)', min_value=0.00, max_value=500.000, value=74.0)
    with col1:
       MDVP = st.number_input('MDVP:Jitter(%)', min_value=0.001, max_value=.001000, value=0.0078)
    with col2:
        MDVP = st.number_input('MDVP:Jitter(Abs)', min_value=0.00001, max_value=0.00090, value=0.00002)
    with col3:
        MDVP = st.number_input('MDVP:RAP', min_value=0.001, max_value=0.01000, value=0.0037)
    with col1:
        MDVP = st.number_input('MDVP:PPQ', min_value=0.001, max_value=0.01000, value=0.0071)
    with col2:
        Jitter = st.number_input('Jitter:DDP', min_value=0.001, max_value=0.01000, value=0.007)
    with col3:
        MDVP = st.number_input('MDVP:Shimmer', min_value=0.001, max_value=0.01000, value=0.007)
    with col1:
        MDVP = st.number_input('MDVP:Shimmer(dB)', min_value=0.01, max_value=0.1000, value=0.246)
    with col2:
        Shimmer = st.number_input('Shimmer:APQ3', min_value=0.001, max_value=0.01000, value=0.0152)
    with col3:
        Shimmer = st.number_input('Shimmer:APQ5', min_value=0.001, max_value=0.01000, value=0.4321)
    with col1:
        MDVP = st.number_input('MDVP:APQ', min_value=0.001, max_value=0.01000, value=0.0359)
    with col2:
        Shimmer = st.number_input('Shimmer:DDA', min_value=0.001, max_value=0.01000, value=0.0349)
    with col3:
        NHR = st.number_input('NHR', min_value=0.001, max_value=0.01000, value=0.0247)
    with col1:
        HNR = st.number_input('HNR', min_value=0.01, max_value=55.00, value=21.33)
    with col2:
       RPDE = st.number_input('RPDE', min_value=0.0001, max_value=0.900000, value=0.031673)
    with col3:
        DFA = st.number_input('DFA', min_value=0.0001, max_value=0.900000, value=0.03734)
    with col1:
        D2 = st.number_input('D2', min_value=0.001, max_value=0.900000, value=0.03745)
    with col2:
        PPE = st.number_input('PPE', min_value=0.001, max_value=0.900000, value=0.6593)

if st.button('Predict', key='predict_button_2'):
    features = np.array([[ 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)','MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP','MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5','MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',  'RPDE', 'DFA','D2', 'PPE']])
    prediction = parkinson_model.predict(features)
    probability = parkinson_model.predict_proba(features)[0][1]

    if prediction == 1:
        st.markdown(f'<h4 style="color:red; background-color:#000; size:20px;">The model predicts you <strong>have parkinson</strong> with a probability of {probability:.2f}.</h4>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    else:
        st.markdown(f'<h4 style="color:green; background-color:#000; size:20px">The model predicts you <strong>do not have parkinson</strong> with a probability of {1 - probability:.2f}.</h4>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    

html_temp = """
    <div style="background-color:black; padding:10px"; color:white;>
    <h5 style="color:white; text-align:center;">&copy 2024 Created by: Odeyale Kehinde Musiliudeen </h5>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
        
