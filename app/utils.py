# app/utils.py

from core_utils import *

# -----------------------
# Safety Rule Function
# -----------------------

def simple_rule_flags(x):
    """Check vitals for warning signs and return alerts."""
    alerts = []

    try:
        spo2 = float(x.get("spo2", 98))
        temp = float(x.get("temp_c", 36.8))
        chest_pain = int(x.get("chest_pain", 0))
        short_breath = int(x.get("shortness_of_breath", 0))
    except (ValueError, TypeError):
        spo2, temp, chest_pain, short_breath = 98, 36.8, 0, 0

    # Low oxygen
    if spo2 < 92:
        alerts.append("⚠️ Low oxygen saturation detected (SpO₂ < 92%). Seek urgent medical attention.")
    # High fever
    if temp >= 39:
        alerts.append("🌡️ High fever detected (≥ 39°C). Consider medical care.")
    # Chest pain + breath shortness
    if chest_pain == 1 and short_breath == 1:
        alerts.append("💔 Chest pain + shortness of breath may indicate a serious issue. Seek emergency care immediately.")
    
    return alerts
