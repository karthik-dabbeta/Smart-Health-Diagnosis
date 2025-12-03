# core_utils.py
from typing import Dict, List

SYMPTOMS = [
    "fever", "cough", "chest_pain", "runny_nose", "diarrhea", "vomiting",
    "body_ache", "frequent_urination", "sore_throat", "headache",
    "shortness_of_breath", "fatigue", "rash", "increased_thirst",
    "vision_blur", "loss_of_smell", "nausea"
]

NUMERIC_FEATURES = ["age", "temp_c", "heart_rate", "spo2"]
CATEGORICAL_FEATURES = ["gender"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + SYMPTOMS
TARGET_NAME = "disease"

RECOMMENDATIONS = {
    "Common Cold": ["Rest, fluids, OTC meds if needed."],
    "Flu": ["Hydrate, rest, take paracetamol for fever."],
    "Migraine": ["Rest in a dark room, avoid triggers."],
    "Gastroenteritis": ["Drink ORS, eat light, avoid dairy."],
    "COVID-19": ["Isolate, monitor SpO2, seek care if low."],
    "Allergy": ["Avoid triggers, antihistamines if needed."],
    "Diabetes (possible)": ["Consult doctor for blood sugar test."],
    "Hypertension (possible)": ["Check BP regularly, reduce salt."],
    "Heart Attack Risk": ["Seek emergency medical help immediately!"],
    "Dermatitis": ["Use moisturizers, avoid irritants."]
}
