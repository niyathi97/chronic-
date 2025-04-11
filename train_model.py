# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("improved_chronic_disease_dataset.csv")

# Separate features and target
X = df.drop(columns=["disease", "treatment", "treatment_success_prob"])
y = df["disease"]

# Encode categorical features
gender_encoder = LabelEncoder()
X["gender"] = gender_encoder.fit_transform(X["gender"])

# Encode target labels
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature scaling (optional, not critical for RandomForest but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the model and encoders
joblib.dump(clf, "chronic_disease_model.pkl")
joblib.dump(disease_encoder, "disease_encoder.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
