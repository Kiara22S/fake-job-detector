from fastapi import FastAPI, HTTPException
from backend.schema import JobListing
from sqlmodel import Session
from backend.database import engine, init_db 
import urllib.parse

# 1. Initialize the FastAPI App
app = FastAPI(title="Fake Job Detector API")

# 2. Run Database Setup on Startup
@app.on_event("startup")
def on_startup():
    init_db()
    print("🚀 Database is connected and tables are ready!")

# 3. Existing Health Check 
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "database": "connected",
        "message": "Fake Job Detector API is running smoothly!"
    }

# 4. NEW: Analyze Job Endpoint & Domain Extraction
import urllib.parse

@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    # This is the "Step 3" logic:
    domain = urllib.parse.urlparse(job_data.company_url).netloc
    
    with Session(engine) as session:
        session.add(job_data)
        session.commit()
        session.refresh(job_data)
    
    return {
        "status": "analyzed and stored", 
        "job_id": job_data.id, 
        "extracted_domain": domain  # Verification for Step 3
    }