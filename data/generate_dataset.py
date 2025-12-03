# generate_dataset.py
import pandas as pd
import numpy as np
import random

# ------------------------
# Configuration
# ------------------------
NUM_ROWS = 800  # You can change between 500-1000
DISEASES = [
    "Common Cold","Flu","Migraine","Gastroenteritis","COVID-19",
    "Allergy","Diabetes (possible)","Hypertension (possible)",
    "Heart Attack Risk","Dermatitis"
]

GENDERS = ["Male", "Female", "Other"]

SYMPTOMS = [
    "fever","cough","chest_pain","runny_nose","diarrhea","vomiting",
    "body_ache","frequent_urination","sore_throat","headache",
    "shortness_of_breath","fatigue","rash","increased_thirst",
    "vision_blur","loss_of_smell","nausea"
]

NUMERIC_FEATURES = ["age","temp_c","heart_rate","spo2"]

# ------------------------
# Function to generate row
# ------------------------
def generate_row():
    disease = random.choice(DISEASES)
    gender = random.choice(GENDERS)
    age = random.randint(5, 80)
    temp_c = round(random.uniform(36.0, 40.0),1)
    heart_rate = random.randint(60, 110)
    spo2 = random.randint(90, 100)

    # Randomly assign symptoms based on disease
    symptom_row = {sym:0 for sym in SYMPTOMS}
    
    if disease == "Common Cold":
        symptom_row.update({ "cough":1, "runny_nose":1, "sore_throat":1 })
    elif disease == "Flu":
        symptom_row.update({ "fever":1, "cough":1, "fatigue":1, "body_ache":1 })
    elif disease == "Migraine":
        symptom_row.update({ "headache":1, "nausea":1, "vision_blur":1 })
    elif disease == "Gastroenteritis":
        symptom_row.update({ "diarrhea":1, "vomiting":1, "nausea":1 })
    elif disease == "COVID-19":
        symptom_row.update({ "fever":1, "cough":1, "shortness_of_breath":1, "loss_of_smell":1 })
    elif disease == "Allergy":
        symptom_row.update({ "runny_nose":1, "rash":1, "cough":1 })
    elif disease == "Diabetes (possible)":
        symptom_row.update({ "frequent_urination":1, "increased_thirst":1 })
    elif disease == "Hypertension (possible)":
        symptom_row.update({ "headache":1, "chest_pain":1 })
    elif disease == "Heart Attack Risk":
        symptom_row.update({ "chest_pain":1, "shortness_of_breath":1 })
    elif disease == "Dermatitis":
        symptom_row.update({ "rash":1 })

    return {
        "age": age,
        "gender": gender,
        "temp_c": temp_c,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "disease": disease,
        **symptom_row
    }

# ------------------------
# Generate dataset
# ------------------------
data = [generate_row() for _ in range(NUM_ROWS)]
df = pd.DataFrame(data)

# Save CSV
df.to_csv("data/symptoms_disease.csv", index=False)
print("✅ Dataset saved as 'data/symptoms_disease.csv'")
