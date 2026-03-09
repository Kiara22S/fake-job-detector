from fastapi import FastAPI, HTTPException
from backend.schema import JobListing
from sqlmodel import Session
from backend.database import engine, init_db
import urllib.parse
import whois
import ssl
import socket
import joblib
import pandas as pd
import os
from datetime import datetime
from scipy.sparse import hstack

# --- IMPORT HER LOGIC ---
# Ensure feature_engineer.py is in the same folder or python path
from feature_engineer import extract_features

# 1. Initialize App & Load ML Assets
app = FastAPI(title="Fake Job Detector - AI Production")

MODEL_PATH = "model/model.pkl"
TFIDF_PATH = "model/tfidf.pkl"

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
tfidf = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None

@app.on_event("startup")
def on_startup():
    init_db()
    print(f"🚀 System Ready. Model: {model is not None}, TFIDF: {tfidf is not None}")

# 2. Security Intel Function (Your Logic)
def get_security_intel(url: str):
    try:
        domain = urllib.parse.urlparse(url).netloc
        if not domain: return "Unknown", None, False
        
        creation_date = None
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list): creation_date = creation_date[0]
        except: pass

        is_ssl_valid = False
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=3) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    ssock.getpeercert()
                    is_ssl_valid = True
        except: is_ssl_valid = False

        return domain, creation_date, is_ssl_valid
    except: return "Error", None, False

# 3. Merged Analyze Endpoint
@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    # Step A: Your Security Intelligence
    domain, created_at, ssl_status = get_security_intel(job_data.company_url)
    
    # Step B: Prepare DataFrame for Kiara's Feature Engineering
    # We use getattr to prevent errors if fields are missing in your schema
    input_df = pd.DataFrame([{
        "title": job_data.title,
        "description": job_data.description,
        "company_profile": getattr(job_data, "company_profile", ""),
        "location": getattr(job_data, "location", ""),
        "requirements": getattr(job_data, "requirements", "")
    }])

    # Step C: Run Her Feature Engineering
    processed_df = extract_features(input_df)
    risk_score = processed_df["risk_score"].iloc[0]

    # Step D: ML Inference (Text + Structured)
    prediction = "Unverified"
    if model and tfidf:
        # 1. Transform Text
        full_text = f"{job_data.title} {job_data.description} {getattr(job_data, 'requirements', '')}"
        X_text = tfidf.transform([full_text])

        # 2. Extract Structured Features (Matches her train.py Step 5)
        features_list = [
            "gmail_domain", "has_payment_request", "contains_urgent_words",
            "salary_mentioned", "location_missing", "description_length",
            "risk_score", "new_domain"
        ]
        X_structured = processed_df[features_list].values

        # 3. Predict
        X_final = hstack([X_text, X_structured])
        pred_bin = model.predict(X_final)[0]
        prediction = "Fake" if pred_bin == 1 else "Real"

    # Step E: Persistence
    with Session(engine) as session:
        job_data.ml_risk_score = float(risk_score)
        job_data.is_fake = (prediction == "Fake")
        session.add(job_data)
        session.commit()
        session.refresh(job_data)

    # Step F: Return Unified Result
    return {
        "job_id": job_data.id,
        "verdict": prediction,
        "final_risk_score": risk_score,
        "intelligence": {
            "domain_age_days": (datetime.now(created_at.tzinfo) - created_at).days if isinstance(created_at, datetime) else 0,
            "ssl_valid": ssl_status,
            "extracted_features": {
                "payment_request": bool(processed_df["has_payment_request"].iloc[0]),
                "urgent_keywords": bool(processed_df["contains_urgent_words"].iloc[0])
            }
        }
    }