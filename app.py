import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Risk Checker",
    page_icon="🚗",
    layout="centered",
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; }
    .title  { font-size: 2.2rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0; }
    .sub    { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .result-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .smart  { background: #d4edda; color: #155724; border-left: 6px solid #28a745; }
    .fair   { background: #fff3cd; color: #856404; border-left: 6px solid #ffc107; }
    .over   { background: #f8d7da; color: #721c24; border-left: 6px solid #dc3545; }
    .avoid  { background: #f5c6cb; color: #491217; border-left: 6px solid #a71d2a; }
    .detail { background: #f8f9fa; border-radius: 10px; padding: 1rem 1.5rem; margin-top: 1rem; font-size: 0.95rem; }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton > button:hover { background-color: #16213e; }
</style>
""", unsafe_allow_html=True)


# ── Download large model from Google Drive ─────────────────────────────────────
@st.cache_resource
def load_models():
    # ------------------------------------------------------------------ #
    #  IMPORTANT: Replace the FILE_ID below with YOUR actual Google Drive  #
    #  file ID for risk_classifier_clean.pkl                               #
    #  It looks like: 1A2B3C4D5E6F7G8H9I0J (from the shareable link)      #
    # ------------------------------------------------------------------ #
    FILE_ID   = "1XS95yPWjmKekx6MBCWZIyoN5cQqBh4qG"
    MODEL_PATH = "risk_classifier_clean.pkl"

    if not os.path.exists(MODEL_PATH):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Could not download the risk model: {e}")
            st.stop()

    price_model      = joblib.load("best_price_model.pkl")
    risk_classifier  = joblib.load(MODEL_PATH)
    feature_cols     = joblib.load("feature_cols.pkl")
    class_features   = joblib.load("classification_features_no_leakage.pkl")
    return price_model, risk_classifier, feature_cols, class_features


# ── Helper functions (same logic as your notebook) ────────────────────────────
def mileage_risk_category(odometer):
    if odometer < 50000:   return "low mileage"
    if odometer < 100000:  return "moderate mileage"
    if odometer < 150000:  return "high mileage"
    return "very high mileage"

def age_risk_category(vehicle_age):
    if vehicle_age <= 5:   return "newer used car"
    if vehicle_age <= 10:  return "moderately aged"
    if vehicle_age <= 15:  return "older car"
    return "very old car"

def title_risk_category(title_status):
    if title_status == "clean":                          return "low title risk"
    if title_status in ["rebuilt", "salvage"]:           return "high title risk"
    if title_status in ["lien", "missing", "parts only"]:return "very high title risk"
    return "unknown title risk"

def predict_buyer_risk(car_info, price_model, risk_clf, feature_cols, class_features):
    df = pd.DataFrame([car_info])

    # normalise text fields
    for col in ["manufacturer","condition","cylinders","fuel",
                "title_status","transmission","drive","type","state"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["vehicle_age"]   = 2026 - df["year"]
    df["mileage_risk"]  = df["odometer"].apply(mileage_risk_category)
    df["age_risk"]      = df["vehicle_age"].apply(age_risk_category)
    df["title_risk"]    = df["title_status"].apply(title_risk_category)

    df["expected_price"] = np.maximum(price_model.predict(df[feature_cols]), 0)
    df["price_gap"]      = df["price"] - df["expected_price"]
    df["price_gap_percent"] = (
        df["price_gap"] / df["expected_price"].replace(0, np.nan) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    prediction   = risk_clf.predict(df[class_features])[0]
    probabilities = risk_clf.predict_proba(df[class_features])
    labels        = list(risk_clf.classes_)
    smart_prob    = probabilities[:, labels.index("smart buy")][0]

    final = prediction
    if final == "smart buy" and smart_prob < 0.75:
        final = "fair deal"

    return {
        "final_category":    final,
        "smart_probability": round(smart_prob, 3),
        "actual_price":      round(df.loc[0, "price"], 2),
        "expected_price":    round(df.loc[0, "expected_price"], 2),
        "price_gap":         round(df.loc[0, "price_gap"], 2),
        "price_gap_percent": round(df.loc[0, "price_gap_percent"], 2),
        "vehicle_age":       int(df.loc[0, "vehicle_age"]),
        "mileage_risk":      df.loc[0, "mileage_risk"],
        "age_risk":          df.loc[0, "age_risk"],
        "title_risk":        df.loc[0, "title_risk"],
    }


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🚗 Used Car Risk Checker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Enter the details of a car listing to see if it\'s a smart buy or a risk.</p>', unsafe_allow_html=True)

price_model, risk_clf, feature_cols, class_features = load_models()

with st.form("car_form"):
    st.subheader("Car Details")

    col1, col2 = st.columns(2)

    with col1:
        price        = st.number_input("Listing Price ($)",        min_value=500,   max_value=150000, value=14000, step=500)
        year         = st.number_input("Year",                     min_value=1980,  max_value=2026,   value=2016,  step=1)
        odometer     = st.number_input("Mileage (miles)",          min_value=0,     max_value=400000, value=85000, step=1000)
        manufacturer = st.selectbox("Manufacturer", sorted([
            "acura","audi","bmw","buick","cadillac","chevrolet","chrysler",
            "dodge","ford","gmc","honda","hyundai","infiniti","jeep","kia",
            "lexus","lincoln","mazda","mercedes-benz","mitsubishi","nissan",
            "ram","subaru","tesla","toyota","volkswagen","volvo","other"
        ]))

    with col2:
        condition    = st.selectbox("Condition",     ["good","excellent","like new","fair","salvage","unknown"])
        fuel         = st.selectbox("Fuel Type",     ["gas","diesel","hybrid","electric","other"])
        transmission = st.selectbox("Transmission",  ["automatic","manual","other"])
        title_status = st.selectbox("Title Status",  ["clean","rebuilt","salvage","lien","missing","parts only"])

    col3, col4 = st.columns(2)
    with col3:
        drive    = st.selectbox("Drive",        ["fwd","rwd","4wd"])
        car_type = st.selectbox("Vehicle Type", ["sedan","SUV","truck","coupe","hatchback","wagon","van","pickup","convertible","other"])
    with col4:
        cylinders = st.selectbox("Cylinders",   ["4 cylinders","6 cylinders","8 cylinders","other"])
        state     = st.selectbox("State (2-letter)", sorted([
            "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id",
            "il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms",
            "mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok",
            "or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv",
            "wi","wy"
        ]))

    submitted = st.form_submit_button("Check This Car →")


# ── Result ─────────────────────────────────────────────────────────────────────
if submitted:
    car_info = {
        "price":               price,
        "year":                year,
        "odometer":            odometer,
        "manufacturer":        manufacturer,
        "condition":           condition,
        "cylinders":           cylinders,
        "fuel":                fuel,
        "title_status":        title_status,
        "transmission":        transmission,
        "drive":               drive,
        "type":                car_type,
        "state":               state,
        "lat":                 39.5,   # approximate US centre
        "long":                -98.35,
        "vin_is_fake_or_unknown": False,
    }

    with st.spinner("Analysing this listing..."):
        try:
            r = predict_buyer_risk(car_info, price_model, risk_clf, feature_cols, class_features)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # pick colour class
    cat = r["final_category"].lower()
    css = "smart" if "smart" in cat else "fair" if "fair" in cat else "over" if "over" in cat else "avoid"

    icons = {"smart": "✅", "fair": "⚠️", "over": "❌", "avoid": "🚫"}
    icon  = icons.get(css, "📊")

    st.markdown(
        f'<div class="result-box {css}">'
        f'{icon} Verdict: <span style="text-transform:capitalize">{r["final_category"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    gap_sign = "+" if r["price_gap"] >= 0 else ""

    st.markdown(f"""
<div class="detail">
    <b>💰 Listing Price:</b> ${r['actual_price']:,.0f}<br>
    <b>📊 Model's Expected Price:</b> ${r['expected_price']:,.0f}<br>
    <b>📉 Price Difference:</b> {gap_sign}${r['price_gap']:,.0f} ({gap_sign}{r['price_gap_percent']:.1f}%)<br><br>
    <b>🗓 Vehicle Age:</b> {r['vehicle_age']} years &nbsp;|&nbsp;
    <b>🛣 Mileage Risk:</b> {r['mileage_risk']} &nbsp;|&nbsp;
    <b>📋 Title Risk:</b> {r['title_risk']}
</div>
""", unsafe_allow_html=True)

    st.caption("This tool is for educational purposes and does not constitute financial advice.")
