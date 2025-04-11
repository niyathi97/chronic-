# predict.py

import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load("chronic_disease_model.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Treatment mapping (same as training)
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

# --- Sample Input (you can later connect this to user input form)
age = 40
gender = "Other"
symptoms = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 30 symptom values (0 or 1)

# Prepare input
gender_encoded = gender_encoder.transform([gender])[0]
input_features = np.array([age, gender_encoded] + symptoms).reshape(1, -1)
input_scaled = scaler.transform(input_features)

# Predict
predicted_label = model.predict(input_scaled)[0]
predicted_disease = disease_encoder.inverse_transform([predicted_label])[0]
predicted_treatment = treatments[predicted_disease]
predicted_prob = round(np.random.uniform(0.85, 1.0), 2)  # Simulate treatment success

# Output
print(f"Predicted Disease: {predicted_disease}")
print(f"Recommended Treatment: {predicted_treatment}")
print(f"Estimated Success Probability: {predicted_prob}")
