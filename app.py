import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")
st.write("This app predicts whether a person has diabetes or not based on their medical data using a Random Forest Classifier")

# Input fields for user to enter medical data 

st.sidebar.header("Enter medical data ")

Pregnancies              = st.sidebar.number_input("Number of Pregnancies" , min_value = 0.0 , max_value = 30.0 , value = 1.0 , step = 1.0)
Glucose                  = st.sidebar.number_input("Glucose Level" , min_value = 1.0 , max_value = 800.0 , value = 120.0 , step = 1.0)
BloodPressure            = st.sidebar.number_input("Blood Pressure" , min_value = 1.0 , max_value = 200.0 ,  value = 70.0 , step = 1.0)
SkinThickness            = st.sidebar.number_input("Skin Thickness" , min_value = 1.0 , max_value = 300.0 ,  value = 20.0 , step = 1.0)
Insulin                  = st.sidebar.number_input("Insulin Level" , min_value = 1.0 , max_value = 1000.0 , value = 80.0 , step = 1.0)
BMI                      = st.sidebar.number_input("Body Mass Index (BMI)" , min_value = 1.0 , max_value = 300.0 ,  value =25.0 , step = 0.1)
DiabetesPedigreeFunction = st.sidebar.number_input("Diabetes Pedigree Function" , min_value = 0.01 , max_value = 2.5 , value = 0.5 , step = 0.01)
Age                      = st.sidebar.number_input("Age" , min_value = 1.0 , max_value = 120.0 , value = 30.0 ,  step = 1.0)

input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin ,
                             BMI, DiabetesPedigreeFunction, Age]], columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                                      "BMI", "DiabetesPedigreeFunction", "Age"])


if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1 :
        st.write("The person is likely to have diabetes")
    else: 
        st.write("The person is unlikely to have diabetes")
    st.subheader("Prediction Result")
    


st.subheader("Your Input")
st.dataframe(pd.DataFrame({
    "Pregnancies" : [Pregnancies] ,
    "Glucose" : [Glucose] ,
    "BloodPressure" : [BloodPressure] ,
    "SkinThickness" : [SkinThickness] ,
    "Insulin" : [Insulin] ,
    "BMI" : [BMI] ,
    "DiabetesPedigreeFunction" : [DiabetesPedigreeFunction] ,
    "Age" : [Age]
}))