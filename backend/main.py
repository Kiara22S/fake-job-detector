from fastapi import FastAPI, HTTPException, Request
from backend.schema import JobListing
from sqlmodel import Session, select
from backend.database import engine, init_db
import urllib.parse
import whois
import ssl
import socket
import joblib
import pandas as pd
import os
import logging
import time
from datetime import datetime
from scipy.sparse import hstack

# --- Import ML Logic ---
from src.feature_engineering import extract_features

# 1. SETUP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake Job Detector - Production API")

MODEL_PATH = "model/model.pkl"
TFIDF_PATH = "model/tfidf.pkl" 

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
tfidf = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info(f"🚀 System Startup: Model Loaded={model is not None}, TFIDF Loaded={tfidf is not None}")

# 2. MIDDLEWARE FOR REQUEST LOGGING
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Duration: {process_time:.4f}s | Status: {response.status_code}")
    return response

def get_security_intel(url: str):
    try:
        domain = urllib.parse.urlparse(url).netloc
        if not domain: return "Unknown", None, False
        creation_date = None
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list): creation_date = creation_date[0]
        except Exception as e:
            logger.warning(f"WHOIS lookup failed for {domain}: {e}")

        is_ssl_valid = False
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=3) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    ssock.getpeercert()
                    is_ssl_valid = True
        except Exception as e:
            logger.warning(f"SSL check failed for {domain}: {e}")

        return domain, creation_date, is_ssl_valid
    except Exception as e:
        logger.error(f"Critical error in security intel: {e}")
        return "Error", None, False

@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    logger.info(f"Analyzing job: {job_data.title} at {job_data.company}")
    
    # Intelligence Gathering
    domain, created_at, ssl_status = get_security_intel(job_data.company_url)
    
    # Feature Engineering
    input_df = pd.DataFrame([{
        "title": job_data.title,
        "description": job_data.description,
        "company_profile": getattr(job_data, "company_profile", ""),
        "location": getattr(job_data, "location", ""),
        "requirements": getattr(job_data, "requirements", "")
    }])
    processed_df = extract_features(input_df)
    raw_risk = float(processed_df["risk_score"].iloc[0])
    normalized_score = min(raw_risk, 1.0) 

    # ML Inference
    prediction = "Unverified"
    if model and tfidf:
        full_text = f"{job_data.title} {job_data.description} {getattr(job_data, 'requirements', '')}"
        X_text = tfidf.transform([full_text])
        features_list = ["gmail_domain", "has_payment_request", "contains_urgent_words",
                         "salary_mentioned", "location_missing", "description_length",
                         "risk_score", "new_domain"]
        X_structured = processed_df[features_list].values
        X_final = hstack([X_text, X_structured])
        prediction = "Fake" if model.predict(X_final)[0] == 1 else "Real"

    # --- TASK: STORE RISK REPORT IN DB ---
    try:
        with Session(engine) as session:
            job_data.ml_risk_score = normalized_score
            job_data.is_fake = (prediction == "Fake")
            session.add(job_data)
            session.commit()
            session.refresh(job_data)
            logger.info(f"✅ Job report stored in DB with ID: {job_data.id}")
    except Exception as e:
        logger.error(f"❌ Database insertion failed: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    # --- TASK: UPDATE RESPONSE FORMAT ---
    return {
        "job_id": job_data.id,
        "verdict": prediction,
        "risk_level": "LOW" if prediction == "Real" else "HIGH",
        "analysis_report": {
            "confidence_score": f"{int(normalized_score * 100)}%",
            "security_check": {
                "domain": domain,
                "ssl_certified": ssl_status,
                "domain_age_days": (datetime.now(created_at.tzinfo) - created_at).days if isinstance(created_at, datetime) else 0
            },
            "content_flags": {
                "payment_detected": bool(processed_df["has_payment_request"].iloc[0]),
                "urgent_language": bool(processed_df["contains_urgent_words"].iloc[0]),
                "free_email_provider": bool(processed_df["gmail_domain"].iloc[0])
            }
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "log_status": "captured"
        }
    }

@app.get("/history")
def get_history(limit: int = 10):
    with Session(engine) as session:
        statement = select(JobListing).order_by(JobListing.id.desc()).limit(limit)
        results = session.exec(statement).all()
        return {"count": len(results), "records": results}