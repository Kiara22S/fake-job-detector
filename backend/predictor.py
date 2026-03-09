import joblib
import pandas as pd
from scipy.sparse import hstack
import os
from feature_engineering import extract_features

# FAANG-level: Load models once at the module level
MODEL_PATH = "model/model.pkl"
TFIDF_PATH = "model/tfidf.pkl" # Kiara needs to provide this!

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
tfidf = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None

def get_ml_prediction(job_data):
    if not model or not tfidf:
        return "Model/Vectorizer Missing", 0.0, {}

    # 1. Prepare input for feature engineering
    input_df = pd.DataFrame([{
        "title": job_data.title,
        "description": job_data.description,
        "company_profile": getattr(job_data, "company_profile", ""),
        "location": getattr(job_data, "location", ""),
        "requirements": getattr(job_data, "requirements", "")
    }])

    # 2. Run Kiara's Feature Engineering
    processed_df = extract_features(input_df)
    
    # 3. Process Text (TF-IDF) - This matches her training step 4
    full_text = f"{job_data.title} {job_data.description} {getattr(job_data, 'requirements', '')}"
    X_text = tfidf.transform([full_text])

    # 4. Process Structured Features - This matches her training step 5
    features_list = [
        "gmail_domain", "has_payment_request", "contains_urgent_words",
        "salary_mentioned", "location_missing", "description_length",
        "risk_score", "new_domain"
    ]
    X_structured = processed_df[features_list].values

    # 5. Combine and Predict
    X_final = hstack([X_text, X_structured])
    pred_bin = model.predict(X_final)[0]
    
    return ("Fake" if pred_bin == 1 else "Real"), processed_df["risk_score"].iloc[0], processed_df.to_dict('records')[0]