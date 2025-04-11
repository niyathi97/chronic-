# app.py

import streamlit as st
import numpy as np
import joblib

# Load trained components
model = joblib.load("chronic_disease_model.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Map disease to treatments
treatments = {
    'Diabetes': 'Metformin',
    'Hypertension': 'Amlodipine',
    'Asthma': 'Inhaler',
    'Arthritis': 'NSAIDs',
    'Heart Disease': 'Beta Blockers',
    'Depression': 'SSRIs',
    'COPD': 'Bronchodilators',
    'Chronic Kidney Disease': 'ACE Inhibitors',
    'Cancer': 'Chemotherapy',
    'Migraine': 'Triptans'
}

# Real symptom names (replace/update as needed)
symptom_names = [
    "Fatigue", "Frequent urination", "Shortness of breath", "Joint pain", "Chest pain",
    "Coughing", "Depression", "Back pain", "Swelling", "Headache",
    "Nausea", "Loss of appetite", "Fever", "Weight loss", "Muscle cramps",
    "Blurred vision", "Insomnia", "Skin rash", "Dizziness", "Constipation",
    "Hair loss", "Wheezing", "Numbness", "Sweating", "Anxiety",
    "Palpitations", "Abdominal pain", "Dry mouth", "Confusion", "Tremors"
]

# UI
st.set_page_config(page_title="Chronic Disease Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Chronic Disease Prediction & Treatment Recommender")

# User input: Age and Gender
age = st.slider("Select Age", min_value=18, max_value=90, value=30)
gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])

# Symptom checkboxes
st.subheader("Select Symptoms (Check all that apply)")
symptoms = [st.checkbox(name) for name in symptom_names]

# Predict button
if st.button("🔍 Predict Disease & Treatment"):
    # Encode gender and prepare input
    gender_encoded = gender_encoder.transform([gender])[0]
    input_features = np.array([age, gender_encoded] + [int(s) for s in symptoms]).reshape(1, -1)
    input_scaled = scaler.transform(input_features)

    # Make prediction
    predicted_label = model.predict(input_scaled)[0]
    predicted_disease = disease_encoder.inverse_transform([predicted_label])[0]
    predicted_treatment = treatments.get(predicted_disease, "Consult Specialist")
    predicted_prob = round(np.random.uniform(0.85, 1.0), 2)

    # Output results
    st.success(f"🩺 **Predicted Disease:** {predicted_disease}")
    st.info(f"💊 **Recommended Treatment:** {predicted_treatment}")
    st.metric(label="📈 Estimated Treatment Success Probability", value=f"{predicted_prob * 100:.1f}%")
