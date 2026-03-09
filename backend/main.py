from fastapi import FastAPI, HTTPException
from backend.schema import JobListing
from sqlmodel import Session, select
from backend.database import engine, init_db
import urllib.parse
import whois
import ssl
import socket
from datetime import datetime

# 1. Initialize the FastAPI App
app = FastAPI(title="Fake Job Detector API")

# High-risk keywords for the scoring engine
SCAM_KEYWORDS = ["wire transfer", "whatsapp only", "crypto", "no experience", "urgent hire", "package forwarding"]

# 2. Database Initialization on Startup
@app.on_event("startup")
def on_startup():
    init_db()
    print("🚀 Database is connected and tables are ready!")

# 3. Security Utility Functions
def get_security_intel(url: str):
    try:
        domain = urllib.parse.urlparse(url).netloc
        if not domain:
            return "Unknown", None, False

        # WHOIS Lookup
        creation_date = None
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
        except:
            pass

        # SSL Validation
        is_ssl_valid = False
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=3) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    ssock.getpeercert()
                    is_ssl_valid = True
        except:
            is_ssl_valid = False

        return domain, creation_date, is_ssl_valid
    except:
        return "Error", None, False

# 4. Task: Analyze Job (Ingestion & Feature Extraction)
@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    domain, created_at, ssl_status = get_security_intel(job_data.company_url)
    
    with Session(engine) as session:
        session.add(job_data)
        session.commit()
        session.refresh(job_data)
    
    return {
        "job_id": job_data.id,
        "status": "analysis_complete",
        "features": {
            "domain_metadata": {
                "extracted_domain": domain,
                "creation_date": str(created_at) if created_at else "unknown",
                "domain_age_days": (datetime.now(created_at.tzinfo) - created_at).days if isinstance(created_at, datetime) else None
            },
            "security_validation": {
                "ssl_valid": ssl_status,
                "protocol": "https" if ssl_status else "http"
            }
        }
    }

# 5. Task: Risk Score Endpoint (Heuristic Engine)
@app.post("/risk-score/{job_id}")
def calculate_risk(job_id: int):
    with Session(engine) as session:
        job = session.get(JobListing, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        domain, created_at, ssl_status = get_security_intel(job.company_url)
        
        score = 0
        reasons = []

        # Logic A: SSL Check (30% weight)
        if not ssl_status:
            score += 30
            reasons.append("Insecure Connection (No SSL)")

        # Logic B: Domain Age Check (50% weight)
        if isinstance(created_at, datetime):
            age_days = (datetime.now(created_at.tzinfo) - created_at).days
            if age_days < 180:
                score += 50
                reasons.append(f"Suspiciously New Domain ({age_days} days)")
        else:
            score += 40
            reasons.append("Unverifiable Domain History")

        # Logic C: Keyword Analysis (20% weight)
        desc_lower = job.description.lower()
        if any(word in desc_lower for word in SCAM_KEYWORDS):
            score += 20
            reasons.append("Contains High-Risk Recruitment Keywords")

        # Update and Persist Score
        job.ml_risk_score = min(score, 100) / 100
        job.is_fake = job.ml_risk_score > 0.6
        session.add(job)
        session.commit()

        return {
            "job_id": job_id,
            "risk_score_percent": f"{int(job.ml_risk_score * 100)}%",
            "verdict": "FLAGGED AS FRAUDULENT" if job.is_fake else "VERIFIED AS LOW RISK",
            "security_flags": reasons
        }

@app.get("/health")
def health_check():
    return {"status": "online", "database": "connected"}