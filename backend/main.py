from fastapi import FastAPI, HTTPException, Request
from backend.schema import JobListing
from sqlmodel import Session, select
from backend.database import engine, init_db
from backend.ml_service import ml_engine  # Import our new Service
import pandas as pd
import logging
import time
from datetime import datetime

# --- Import Feature Engineering Logic ---
from src.feature_engineering import extract_features

# 1. CONFIGURE LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake Job Detector - v2 Modular API")

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("🚀 System Startup: Infrastructure and DB Initialized")

# 2. MONITORING MIDDLEWARE
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Time: {process_time:.4f}s | Status: {response.status_code}")
    return response

# --- TASK 2: BUILD ENDPOINT TO CALL MODEL (v2) ---
@app.post("/v2/predict")
def predict_only(job_data: JobListing):
    """Dedicated endpoint for pure ML inference without DB storage."""
    try:
        # Feature Engineering
        input_df = pd.DataFrame([job_data.dict()])
        processed_df = extract_features(input_df)
        
        # Call the new ML Service
        verdict, probability = ml_engine.predict(job_data, processed_df)
        
        return {
            "verdict": verdict,
            "model_confidence": f"{int(probability * 100)}%",
            "is_suspicious": probability > 0.7,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="ML Inference failed")

# --- REFACTORED MAIN ANALYZE ENDPOINT ---
@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    if not job_data.company_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    try:
        # 1. Feature Engineering
        input_df = pd.DataFrame([job_data.dict()])
        processed_df = extract_features(input_df)
        
        # 2. TASK 3: INTEGRATE PROBABILITY INTO RISK SCORE
        # We get the AI probability and the manual risk_score
        verdict, ml_prob = ml_engine.predict(job_data, processed_df)
        heuristic_score = float(processed_df["risk_score"].iloc[0])
        
        # Weighted Risk Formula: 70% AI, 30% Rules
        final_risk_score = (0.7 * ml_prob) + (0.3 * min(heuristic_score, 1.0))

        # 3. Persistence
        with Session(engine) as session:
            job_data.ml_risk_score = final_risk_score
            job_data.is_fake = (verdict == "Fake")
            session.add(job_data)
            session.commit()
            session.refresh(job_data)
            logger.info(f"✅ Record {job_data.id} saved with weighted risk: {final_risk_score:.2f}")

        return {
            "job_id": job_data.id,
            "verdict": verdict,
            "risk_level": "LOW" if (verdict == "Real" and final_risk_score < 0.4) else "HIGH",
            "analysis_report": {
                "combined_confidence": f"{int(final_risk_score * 100)}%",
                "ml_probability": f"{int(ml_prob * 100)}%",
                "heuristic_score": f"{int(min(heuristic_score, 1.0) * 100)}%"
            },
            "metadata": {"timestamp": datetime.now().isoformat(), "status": "stored"}
        }

    except Exception as e:
        logger.critical(f"Pipeline failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/history")
def get_history(limit: int = 10):
    with Session(engine) as session:
        statement = select(JobListing).order_by(JobListing.id.desc()).limit(limit)
        results = session.exec(statement).all()
        return {"count": len(results), "records": results}

from sqlmodel import func

@app.get("/stats")
def get_dashboard_stats():
    with Session(engine) as session:
        # 1. Total Count
        total_jobs = session.scalar(select(func.count(JobListing.id)))
        
        # 2. Fake vs Real Count
        fake_count = session.scalar(select(func.count(JobListing.id)).where(JobListing.is_fake == True))
        real_count = total_jobs - fake_count if total_jobs else 0
        
        # 3. Average Risk Score
        avg_risk = session.scalar(select(func.avg(JobListing.ml_risk_score))) or 0.0
        
        # 4. Success Rate (Percentage of real jobs)
        accuracy_rate = (real_count / total_jobs * 100) if total_jobs else 0
        
        return {
            "summary": {
                "total_scanned": total_jobs,
                "fakes_blocked": fake_count,
                "legit_jobs": real_count
            },
            "analytics": {
                "system_average_risk": f"{int(avg_risk * 100)}%",
                "platform_health_score": f"{int(accuracy_rate)}%"
            },
            "timestamp": datetime.now().isoformat()
        }        