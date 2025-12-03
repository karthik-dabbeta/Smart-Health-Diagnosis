# .venv\Scripts\activate
# python train_model.py
# streamlit run app/app_streamlit.py

# app/app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import os
import sys
import json
import datetime
import hashlib
from fpdf import FPDF

# ========== PATH CONFIG ==========
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    import core_utils as cu
except Exception:
    st.error("⚠️ Missing 'core_utils.py' in project root. Please add it and restart.")
    st.stop()

st.set_page_config(page_title="Health Dashboard", layout="wide", initial_sidebar_state="expanded")

MODEL_DIR = os.path.join(ROOT, "models")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
DT_PATH = os.path.join(MODEL_DIR, "dt_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.joblib")
USERS_FILE = os.path.join(ROOT, "users.json")
REPORTS_DIR = os.path.join(ROOT, "reports")

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

rf_model = joblib.load(RF_PATH)
dt_model = joblib.load(DT_PATH)
model_features = joblib.load(FEATURES_PATH)

SYMPTOMS = cu.SYMPTOMS
NUMERIC_FEATURES = cu.NUMERIC_FEATURES
CATEGORICAL_FEATURES = cu.CATEGORICAL_FEATURES
RECOMMENDATIONS = cu.RECOMMENDATIONS

# ========== UTILS ==========
def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def generate_pdf(username, prediction, vitals, symptoms, model_name, recommendations, out_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Health Diagnosis Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"User: {username}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Predicted Disease: {prediction}", ln=True)
    pdf.cell(0, 8, f"Model Used: {model_name}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Vitals:", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in vitals.items():
        pdf.cell(0, 7, f"- {k}: {v}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Symptoms:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, ", ".join(symptoms) if symptoms else "None")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=11)
    for r in recommendations:
        pdf.multi_cell(0, 7, f"- {r}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pdf.output(out_path)
    return out_path

# ========== STYLE ==========
st.markdown("""
<style>
body {background-color: #f5f8fa;}
.top-header {
    background: linear-gradient(90deg, #eaf8f5, #e8f4fc);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    text-align: center;
    color: #003d3d;
}
.top-header h2 {
    font-size: 32px;
    color: #00796b;
}
.top-header p {
    font-size: 17px;
    color: #222;
}
.nav-container {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 25px;
    flex-wrap: wrap;
}
.stButton > button {
    background: white;
    border: 2px solid #009688;
    border-radius: 12px;
    color: #00695c;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: #009688;
    color: white;
    transform: scale(1.07);
}
.report-box {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ========== SESSION ==========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "home"

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🔐 Login / Signup")
    if not st.session_state.logged_in:
        mode = st.radio("Select Mode", ["Login", "Signup"], horizontal=True)
        if mode == "Signup":
            su_user = st.text_input("Username")
            su_email = st.text_input("Email (optional)")
            su_pwd = st.text_input("Password", type="password")
            su_pwd2 = st.text_input("Confirm Password", type="password")
            if st.button("Create Account"):
                users = load_users()
                if not su_user.strip() or not su_pwd:
                    st.error("Username and password required.")
                elif su_user in users:
                    st.error("Username already exists.")
                elif su_pwd != su_pwd2:
                    st.error("Passwords do not match.")
                else:
                    users[su_user] = {"password": hash_password(su_pwd), "email": su_email, "history": []}
                    save_users(users)
                    st.success("✅ Account created! Switch to Login to continue.")
        else:
            li_user = st.text_input("Username")
            li_pwd = st.text_input("Password", type="password")
            if st.button("Login"):
                users = load_users()
                if li_user in users and users[li_user]["password"] == hash_password(li_pwd):
                    st.session_state.logged_in = True
                    st.session_state.username = li_user
                    st.session_state.page = "home"
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")
    else:
        st.write(f"👤 Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()

# ========== PUBLIC VIEW ==========
if not st.session_state.logged_in:
    st.markdown("""
    <div class="top-header">
        <h2>💚 Smart Health Diagnosis</h2>
        <p>Welcome to <b>Smart Health Diagnosis</b> — your intelligent medical assistant that helps analyze your symptoms, 
        predict possible diseases, and provide actionable health recommendations.  
        Please login or sign up using the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ========== HEADER ==========
st.markdown(f"""
<div class="top-header">
    <h2>💚 Welcome, {st.session_state.username}</h2>
    <p>Your personalized AI Health Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ========== NAVBAR ==========
st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
cols = st.columns(6)
nav_items = [
    ("🏠", "Home", "home"),
    ("🩺", "Diagnosis", "diagnosis"),
    ("💊", "Medicines", "medicine"),
    ("👤", "Profile", "profile"),
    ("🤖", "Chat", "chat"),
    ("📜", "History", "history"),
    ("💬", "Feedback", "feedback")  # <— add this line
]

nav_cols = st.columns(len(nav_items))
for i, (icon, label, page) in enumerate(nav_items):
    with nav_cols[i]:
        if st.button(f"{icon} {label}", key=f"nav_{page}"):
            st.session_state.page = page

# ========== PAGE CONTENT ==========
page = st.session_state.page

if page == "home":
    st.write("""
    ### Home
    Welcome to your **Smart Health Dashboard**.  
    - Analyze symptoms using AI  
    - Search for medicines  
    - View history and health reports  
    """)

elif page == "diagnosis":
    st.header("🩺 Symptom Checker")

    # ---- VITALS INPUT ----
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 25, key="age_input")
        temp_c = st.number_input("Temperature (°C)", 30.0, 45.0, 36.5, key="temp_input")
        heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 75, key="hr_input")
    with col2:
        spo2 = st.number_input("SpO2 (%)", 50, 100, 98, key="spo2_input")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_input")

    # ---- SYMPTOMS ----
    st.subheader("Select Symptoms:")
    cols = st.columns(4)
    selected = []
    for i, sym in enumerate(SYMPTOMS):
        if cols[i % 4].checkbox(sym.replace("_", " ").title(), key=f"symptom_{i}"):
            selected.append(sym)

    # ---- MODEL ----
    model_choice = st.radio("Select Model", ["Random Forest", "Decision Tree"], key="model_choice")
    model = rf_model if model_choice == "Random Forest" else dt_model

    # ---- PREDICT ----
    if st.button("Predict", key="predict_btn"):
        try:
            # Collect all data
            data = {
                "age": age,
                "temp_c": temp_c,
                "heart_rate": heart_rate,
                "spo2": spo2,
                "gender": gender
            }
            for s in SYMPTOMS:
                data[s] = 1 if s in selected else 0

            # Reorder columns to match model
            df = pd.DataFrame([data])[model_features]
            pred = model.predict(df)[0]

            # ---- Show results ----
            st.success(f"✅ Predicted Disease: **{pred}**")

            recs = RECOMMENDATIONS.get(pred, ["Consult a doctor for confirmation."])
            st.subheader("💡 Health Recommendations:")
            for r in recs:
                st.write(f"- {r}")

            # ---- Save to user history ----
            users = load_users()
            user = users.get(st.session_state.username, {})
            user.setdefault("history", []).append({
                "timestamp": str(datetime.datetime.now()),
                "prediction": pred,
                "symptoms": selected,
                "model": model_choice,
                "vitals": data
            })
            users[st.session_state.username] = user
            save_users(users)

            # ---- Generate PDF ----
            os.makedirs(REPORTS_DIR, exist_ok=True)
            pdf_path = os.path.join(
                REPORTS_DIR, 
                f"{st.session_state.username}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            generate_pdf(
                username=st.session_state.username,
                prediction=pred,
                vitals=data,
                symptoms=selected,
                model_name=model_choice,
                recommendations=recs,
                out_path=pdf_path
            )

            # ---- Download Button ----
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Diagnosis Report",
                    data=pdf_file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key="download_pdf"
                )

        except Exception as e:
            st.error(f"Prediction Error: {e}")


elif page == "medicine":
    st.header("💊 Medicine Search")
    query = st.text_input("Enter a symptom or condition:")
    if st.button("Search"):
        meds = {"fever": ["Paracetamol"], "cold": ["Cetirizine"], "cough": ["Dextromethorphan"]}
        for m in meds.get(query.lower(), ["No medicine found."]):
            st.write(f"- {m}")

elif page == "profile":
    st.header("👤 My Profile")
    users = load_users()
    u = users.get(st.session_state.username, {})
    st.write(f"**Username:** {st.session_state.username}")
    st.write(f"**Email:** {u.get('email', 'Not provided')}")
    st.write(f"**Total Predictions:** {len(u.get('history', []))}")

elif page == "chat":
    st.header("🤖 AI Health Assistant")
    q = st.text_area("Ask something:")
    if st.button("Ask"):
        if "fever" in q.lower():
            st.write("It might be a viral fever. Drink fluids and rest.")
        else:
            st.write("Consult a doctor for accurate diagnosis.")

elif page == "history":
    st.header("📜 Prediction History")
    users = load_users()
    hist = users.get(st.session_state.username, {}).get("history", [])
    if hist:
        for rec in reversed(hist[-10:]):
            st.write(f"🕒 {rec['timestamp']} → **{rec['prediction']}** ({rec['model']})")
            st.write(f"Symptoms: {', '.join(rec['symptoms'])}")
            st.markdown("---")
    else:
        st.info("No records yet.")
# ---------- FEEDBACK ----------
elif page == "feedback":
    st.header("💬 Feedback")
    st.markdown("We value your input! Please share your thoughts or suggestions below to help us improve the Smart Health Dashboard.")
    
    feedback_text = st.text_area("Your Feedback:", key="feedback_input", placeholder="Type your feedback here...")
    
    if st.button("Submit Feedback", key="submit_feedback"):
        if feedback_text.strip() == "":
            st.warning("⚠️ Please enter feedback before submitting.")
        else:
            feedback_entry = f"{datetime.datetime.now()} - {st.session_state.username}: {feedback_text}\n"
            feedback_path = os.path.join(ROOT, "feedback.txt")
            
            with open(feedback_path, "a", encoding="utf-8") as f:
                f.write(feedback_entry)
            
            st.success("✅ Thank you for your valuable feedback!")

    # Optionally show recent feedbacks for logged-in user
    st.subheader("📝 Your Recent Feedback")
    feedback_path = os.path.join(ROOT, "feedback.txt")
    if os.path.exists(feedback_path):
        with open(feedback_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if st.session_state.username in line]
            recent = lines[-5:] if lines else []
            if recent:
                for fb in reversed(recent):
                    st.write(f"💭 {fb}")
            else:
                st.info("No feedback submitted yet.")
    else:
        st.info("No feedback file found yet.")


st.markdown("---")
st.caption("© 2025 Smart Health Dashboard")
