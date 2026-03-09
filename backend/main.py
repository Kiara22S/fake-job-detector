from fastapi import FastAPI, HTTPException
from backend.schema import JobListing
from sqlmodel import Session
from backend.database import engine, init_db
import urllib.parse
import whois
import ssl
import socket
from datetime import datetime

# 1. Initialize the FastAPI App
app = FastAPI(title="Fake Job Detector API")

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

# 4. The Core Analyze Endpoint with Structured JSON Output
@app.post("/analyze-job")
def analyze_job(job_data: JobListing):
    # Step 1: Extract Intel (Domain, WHOIS, SSL)
    domain, created_at, ssl_status = get_security_intel(job_data.company_url)
    
    # Step 2: Store in Postgres
    with Session(engine) as session:
        session.add(job_data)
        session.commit()
        session.refresh(job_data)
    
    # Step 3: Return Structured JSON Features (Task Complete)
    # This structure is optimized for ML and Frontend consumption
    return {
        "job_id": job_data.id,
        "status": "analysis_complete",
        "features": {
            "domain_metadata": {
                "extracted_domain": domain,
                "creation_date": str(created_at) if created_at else "unknown",
                # Replace line 73 with this:
                "domain_age_days": (datetime.now(created_at.tzinfo) - created_at).days if isinstance(created_at, datetime) else None
            },
            "security_validation": {
                "ssl_valid": ssl_status,
                "protocol": "https" if ssl_status else "http"
            },
            "content_summary": {
                "title": job_data.title,
                "company": job_data.company,
                "desc_length": len(job_data.description)
            }
        },
        "ml_ready": True
    }

# 5. Health Check
@app.get("/health")
def health_check():
    return {"status": "online", "database": "connected"}