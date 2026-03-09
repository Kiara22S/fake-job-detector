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

# --- IMPORT FEATURE ENGINEERING ---
from src.feature_engineering import extract_features

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

# 2. Security Intel Function (SSL/WHOIS)
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

# 3. Merged Analyze & Store Endpoint
@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    # Step A: Security Intelligence
    domain, created_at, ssl_status = get_security_intel(job_data.company_url)
    
    # Step B: Prepare ML Input
    input_df = pd.DataFrame([{
        "title": job_data.title,
        "description": job_data.description,
        "company_profile": getattr(job_data, "company_profile", ""),
        "location": getattr(job_data, "location", ""),
        "requirements": getattr(job_data, "requirements", "")
    }])

    # Step C: Run Feature Engineering
    processed_df = extract_features(input_df)
    risk_score = float(processed_df["risk_score"].iloc[0])

    # Step D: ML Prediction
    prediction = "Unverified"
    if model and tfidf:
        full_text = f"{job_data.title} {job_data.description} {getattr(job_data, 'requirements', '')}"
        X_text = tfidf.transform([full_text])

        features_list = [
            "gmail_domain", "has_payment_request", "contains_urgent_words",
            "salary_mentioned", "location_missing", "description_length",
            "risk_score", "new_domain"
        ]
        X_structured = processed_df[features_list].values
        X_final = hstack([X_text, X_structured])
        
        pred_bin = model.predict(X_final)[0]
        prediction = "Fake" if pred_bin == 1 else "Real"

    # --- YOUR TASK: Store Risk Report in DB ---
    with Session(engine) as session:
        job_data.ml_risk_score = risk_score
        job_data.is_fake = (prediction == "Fake")
        # SSL and Domain info are kept in session for the report
        session.add(job_data)
        session.commit()
        session.refresh(job_data)

    # Step E: Return Unified Result
    return {
        "job_id": job_data.id,
        "status": "stored_in_db",
        "verdict": prediction,
        "risk_report": {
            "score": f"{int(risk_score * 100)}%",
            "ssl_valid": ssl_status,
            "domain": domain,
            "domain_age_days": (datetime.now(created_at.tzinfo) - created_at).days if isinstance(created_at, datetime) else 0
        }
    }