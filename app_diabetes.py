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
diabetes_model = joblib.load('diabetes_model_new2.pkl')

 


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
    probability = diabetes_model.predict_proba(features)[0][1]

    if prediction == 1:
        st.markdown(f'<h4 style="color:red; background-color:#000; size:20px;">The model predicts you <strong>have diabetes</strong> with a probability of {probability:.2f}.</h4>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    else:
        st.markdown(f'<h4 style="color:green; background-color:#000; size:20px">The model predicts you <strong>do not have diabetes</strong> with a probability of {1 - probability:.2f}.</h4>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    
     
 
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    
    html_temp = """
    <div style="background-color:purple; padding:10px">
    <h2 style="color:white; text-align:center;">Heart Disease Prediction using Machine Learning </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:red; background-color:#000; size:20px;">The model is under construction.</h2>', unsafe_allow_html=True)

    
             
# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
   
    html_temp = """
    <div style="background-color:brown; padding:10px">
    <h2 style="color:white; text-align:center;">Parkinson's Disease Prediction using ML </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:red; background-color:#000; size:20px;">The model is under construction.</h2>', unsafe_allow_html=True) 

html_temp = """
    <div style="background-color:black; padding:10px"; color:white;>
    <h5 style="color:white; text-align:center;">&copy 2024 Created by: Odeyale Kehinde Musiliudeen </h5>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
        
