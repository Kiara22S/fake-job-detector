from fastapi import FastAPI
from backend.database import init_db

# 1. Initialize the FastAPI App
app = FastAPI(title="Fake Job Detector API")

# 2. Run Database Setup on Startup
@app.on_event("startup")
def on_startup():
    init_db()
    print("🚀 Database is connected and tables are ready!")

# 3. Create the /health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "database": "connected",
        "message": "Fake Job Detector API is running smoothly!"
    }