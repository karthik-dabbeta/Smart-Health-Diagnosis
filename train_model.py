# train_model.py
# Run this after updating your dataset: python train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/symptoms_disease.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

# Ensure proper columns
TARGET = "disease"
NUMERIC_FEATURES = ["age", "temp_c", "heart_rate", "spo2"]
CATEGORICAL_FEATURES = ["gender"]
SYMPTOMS = [
    "fever", "cough", "chest_pain", "runny_nose", "diarrhea", "vomiting",
    "body_ache", "frequent_urination", "sore_throat", "headache",
    "shortness_of_breath", "fatigue", "rash", "increased_thirst",
    "vision_blur", "loss_of_smell", "nausea"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + SYMPTOMS

# Ensure all required columns exist
for col in ALL_FEATURES:
    if col not in df.columns:
        df[col] = 0

# Drop missing target rows
df = df.dropna(subset=[TARGET])

# Convert categorical gender to string
df["gender"] = df["gender"].astype(str)

# -------------------------------
# Preprocessing & Split
# -------------------------------
X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
], remainder="passthrough")

# -------------------------------
# Train Models
# -------------------------------
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

dt_model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", DecisionTreeClassifier(max_depth=8, random_state=42))
])

rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# -------------------------------
# Save Models and Feature Order
# -------------------------------
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.joblib"))
joblib.dump(dt_model, os.path.join(MODEL_DIR, "dt_model.joblib"))
joblib.dump(ALL_FEATURES, os.path.join(MODEL_DIR, "model_features.joblib"))

print("✅ Models trained and saved successfully.")
