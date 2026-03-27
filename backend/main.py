from fastapi import FastAPI, HTTPException, Request, Depends  # Add 'Depends'
from fastapi.security import OAuth2PasswordRequestForm       # For Login
from backend.auth import create_access_token, verify_password, get_current_user # Our new service
# ... keep your existing imports below


from fastapi import FastAPI, HTTPException, Request
from backend.schema import JobListing,User
from sqlmodel import Session, select, func
from backend.database import engine, init_db
from backend.ml_service import ml_engine
import pandas as pd
import logging
import time
from datetime import datetime
from src.feature_engineering import extract_features

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake Job Detector - Production Modular API")

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info(f"🚀 System Online | Model Version: {ml_engine.version}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Latency: {process_time:.4f}s")
    return response

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        # Check if user exists in the DB
        user = session.exec(select(User).where(User.username == form_data.username)).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        
        # Create the JWT
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}

@app.post("/analyze-job")
def analyze_job(job_data: JobListing ,current_user: str = Depends(get_current_user)):
    if not job_data.company_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    try:
        input_df = pd.DataFrame([job_data.dict()])
        processed_df = extract_features(input_df)
        
        # 1. We now get 'report' (Kiara's work) from our ml_engine
        verdict, ml_prob, report = ml_engine.predict(job_data, processed_df)
        
        heuristic_score = float(processed_df["risk_score"].iloc[0])
        final_risk = (0.7 * ml_prob) + (0.3 * min(heuristic_score, 1.0))

        with Session(engine) as session:
            job_data.ml_risk_score = final_risk
            job_data.is_fake = (verdict == "Fake")
            job_data.model_version = ml_engine.version
            session.add(job_data)
            session.commit()
            session.refresh(job_data)

        # 2. Return your HIGH-LEVEL JSON structure
        return {
            "success": True,
            "data": {
                "job_id": job_data.id,
                "verdict": verdict,
                "risk_details": report, # <--- Kiara's logic plugged in here
                "scoring": {
                    "combined_confidence": round(final_risk, 4),
                    "ml_probability": round(ml_prob, 4),
                    "heuristic_score": round(min(heuristic_score, 1.0), 4)
                },
                "lineage": {
                    "model_version": ml_engine.version,
                    "engine_type": "RandomForest-Weighted",
                    "processed_at": datetime.now().isoformat()
                }
            },
            "flags": {
                "is_suspicious_domain": "xyz" in job_data.company_url or "top" in job_data.company_url,
                "high_risk_detected": final_risk > 0.7
            }
        }
    except Exception as e:
        logger.critical(f"Analysis failure: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

@app.get("/stats")
def get_dashboard_stats():
    with Session(engine) as session:
        total = session.scalar(select(func.count(JobListing.id))) or 0
        fakes = session.scalar(select(func.count(JobListing.id)).where(JobListing.is_fake == True)) or 0
        return {
            "total_scanned": total,
            "fakes_detected": fakes,
            "current_model": ml_engine.version
        }

@app.get("/history")
def get_history(limit: int = 10 ,current_user: str = Depends(get_current_user)):
    with Session(engine) as session:
        results = session.exec(select(JobListing).order_by(JobListing.id.desc()).limit(limit)).all()
        return {
            "success": True,
            "count": len(results),
            "records": results
        }