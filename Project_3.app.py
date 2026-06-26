from json import encoder
from random import Random
from statistics import LinearRegression

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("Fitness Calorie Predictor App")

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to",["Page 1", "Page 2"])

if page == "Page 1":
    st.header("Supervised Machine Learning")
    try:
        model = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    except FileNotFoundError:
        st.error("Model or Scaler binary files are missing from your project folder.")
        st.stop()
    st.write("Fill in your data and click submit to compute your estimated calorie burn.")

    with st.form("comprehensive_metrics_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=70.0)
            height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.75)
            max_bpm = st.number_input("Max Heart Rate (BPM)", min_value=50, max_value=250, value=175)
            avg_bpm = st.number_input("Average Heart Rate (BPM)", min_value=40, max_value=220, value=140)
            resting_bpm = st.number_input("Resting Heart Rate (BPM)", min_value=30, max_value=150, value=65)
            session_duration = st.number_input("Workout Duration (hours)", min_value=0.1, max_value=24.0, value=1.0)
            gender = st.selectbox("Gender Orientation", ["Male", "Female"])

        with col2:
            fat_percentage = st.number_input("Body Fat Percentage (%)", min_value=1.0, max_value=70.0, value=18.5)
            water_intake = st.number_input("Water Intake (liters)", min_value=0.0, max_value=20.0, value=1.5)
            workout_frequency = st.number_input("Workout Frequency (days/week)", min_value=0, max_value=7, value=3)
            bmi = st.number_input("BMI Value", min_value=10.0, max_value=60.0, value=22.8)
            base_met = st.number_input("Base Metabolic Rate", min_value=0.3, max_value=15.0, value=0.9)
            hr_intensity = st.number_input("HR Intensity", min_value=0.0, max_value=20.0, value=0.9)
            effective_met = st.number_input("Effective MET Value", min_value=1.0, max_value=20.0, value=6.5)
            experience_choice = st.selectbox("Experience Level Status", ["Beginner", "Intermediate", "Advanced"])
            workout_choice = st.selectbox("Workout Classification Type", ["Cardio", "HIIT", "Mixed", "Strength", "Yoga"])

        submit_button = st.form_submit_button(label=" Calculate Calories Burned")

    if submit_button:
    
        if experience_choice == "Beginner":
            experience_encoded = 0
        elif experience_choice == "Intermediate":
            experience_encoded = 1
        else:
            experience_encoded = 2

    
        w_cardio, w_hiit, w_mixed, w_strength, w_yoga = 0, 0, 0, 0, 0
    
        if workout_choice == "Cardio":
            w_cardio = 1
        elif workout_choice == "HIIT":
            w_hiit = 1
        elif workout_choice == "Mixed":
            w_mixed = 1
        elif workout_choice == "Strength":
            w_strength = 1
        elif workout_choice == "Yoga":
            w_yoga = 1

        g_female, g_male = 0, 0
        if gender == "Female":
            g_female = 1
        else:
            g_male = 1

        #  Arrange variables into a single row matrix (EXACT alignment with your 22 features)
        raw_inputs = [[
        age, weight, height, max_bpm, avg_bpm, resting_bpm,
        session_duration, fat_percentage, water_intake,
        workout_frequency, experience_encoded, bmi, base_met,
        hr_intensity, effective_met,
        w_cardio, w_hiit, w_mixed, w_strength, w_yoga,
        g_female, g_male]]
    
        # Transform arrays and predict
        sample_input = np.array(raw_inputs)
        sample_input_scaled = scaler.transform(sample_input)
        prediction = model.predict(sample_input_scaled)
    
        #  Print out result to user screen interface
        st.markdown("---")
        st.success(f" Estimated Calories Burned:**{prediction[0]:.2f} kcal**")
    


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


    
if page == "Page 2":
    st.header("Unsupervised Machine Learning")
    try: 
        kmeans = joblib.load("kmeans_model.joblib")
        scaler = joblib.load("scaler.joblib")
    except FileNotFoundError:
        st.error("Model or Scaler not found in your project folder")
    st.write("Enter the user metrics to prdictic fitness cluster")

    with st.form("comprehensive_metrics_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0)
            height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.90)
            max_bpm = st.number_input("Max BPM", min_value=50, max_value=250, value=180)
            avg_bpm = st.number_input("Avg BPM", min_value=40, max_value=220, value=140)
            resting_bpm = st.number_input("Resting BPM", min_value=30, max_value=150, value=65)
            session_duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=6.0, value=1.0)
            fat_percentage = st.number_input("Fat Percentage", min_value=1.0, max_value=70.0, value=18.5)
            
        
        with col2:
            water_intake = st.number_input("Water Intake (liters)", min_value=0.0, max_value=6.0, value=2.5)
            workout_freq = st.slider("Workout Frequency (days/week)", min_value=0, max_value=7, value=3)
            experience_level = st.slider("Experience Level (1-3)", min_value=1, max_value=3, value=2)
            bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.8)
            base_met = st.number_input("Base Metabolic Rate", min_value=0.3, max_value=15.0, value=0.9)
            hr_intensity = st.number_input("HR Intensity", min_value=0.0, max_value=20.0, value=0.9)                                                               
            effective_met = st.number_input("Effective MET", min_value=0.0, max_value=20.0, value=1.2)
            calories_burned = st.number_input("Calories Burned (kcal)", min_value=0, max_value=5000, value=400)
            gender = st.selectbox("Gender", ["Female", "Male"])

        
            gender_Female = 1 if gender == "Female" else 0
            gender_Male = 1 if gender == "Male" else 0
            
        features_array = np.array(
        [[age, weight,height,max_bpm,avg_bpm,resting_bpm,session_duration,fat_percentage,water_intake,
        workout_freq,experience_level,bmi,base_met,hr_intensity,effective_met,calories_burned,
        gender_Female,gender_Male]])
        
        if st.form_submit_button("Predict Cluster"):
            
            features_scaled = scaler.transform(features_array)

            cluster_num  = kmeans.predict(features_scaled)[0]

            cluster_mapping = {0:"Cardio", 1:"HIIT", 2:"Mixed", 3:"Strength", 4:"Yoga"}

            predicted_id = int(cluster_num)

            cluster_name = cluster_mapping.get(predicted_id, f"Unknow Cluster ({predicted_id})")
        st.success(f" The user belongs to:  ** {cluster_name}**")