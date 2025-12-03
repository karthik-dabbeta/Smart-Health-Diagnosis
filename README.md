# Smart Health Diagnosis System (ML)

A minimal, **college-presentable** project that predicts likely diseases based on symptoms and vitals.
Built with **Python**, **scikit-learn**, and **Streamlit**.

> **Disclaimer:** This app is for educational purposes only and is **not** a substitute for professional medical advice.

## Features

- Symptom checker form (age, gender, vitals, and multiple symptoms)
- ML prediction using **RandomForest** (and a DecisionTree baseline)
- Probability (confidence) scores for each predicted disease
- Rule-based safety recommendations and next steps
- Feedback loop: user can mark prediction as correct/incorrect (saved locally)
- Clean, responsive Streamlit UI

## Tech Stack

- Python, scikit-learn, pandas, numpy
- Streamlit (web UI)
- Joblib for model persistence
- Synthetic dataset (can be replaced with a public dataset later)

## Project Structure

```
smart-health-diagnosis/
├── app/
│   └── app_streamlit.py
├── data/
│   └── symptom_disease.csv
├── models/
│   ├── rf_model.joblib
│   ├── dt_model.joblib
│   └── encoders.joblib
├── train_model.py
├── utils.py
├── requirements.txt
└── README.md
```

## How to Run (Local)

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # on Windows
   # or
   source .venv/bin/activate       # on macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain models:**
   ```bash
   python train_model.py
   ```

4. **Start the app:**
   ```bash
   streamlit run app/app_streamlit.py
   ```

5. Open the local URL Streamlit prints (usually http://localhost:8501).

## Replacing the Dataset

- Replace `data/symptom_disease.csv` with a real dataset (e.g., from Kaggle).
- Make sure to keep the same column names or update `train_model.py` and `utils.py` accordingly.

## Notes

- Feedback submissions are stored in `data/feedback.jsonl` (append-only).
- This is a simplified demo for education. Real deployments must undergo clinical validation.
