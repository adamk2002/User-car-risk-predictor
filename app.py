@st.cache_resource
def load_models():
    price_model     = joblib.load("best_price_model.pkl")
    risk_classifier = joblib.load("risk_classifier_clean.pkl")
    feature_cols    = joblib.load("feature_cols.pkl")
    class_features  = joblib.load("classification_features_no_leakage.pkl")
    return price_model, risk_classifier, feature_cols, class_features
