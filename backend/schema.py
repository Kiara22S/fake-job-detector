from sqlmodel import SQLModel, Field, JSON, Column  # Import JSON and Column
from typing import Optional, Dict, Any

from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    email: str = Field(unique=True)
    password: str

class JobListing(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    company: str
    description: str
    company_url: str = Field(default="https://unknown.com")
    location: Optional[str] = Field(default="")
    company_profile: Optional[str] = Field(default="")
    requirements: Optional[str] = Field(default="")
    
    # AI & Metadata Fields
    ml_risk_score: float = 0.0
    is_fake: bool = False
    model_version: str = Field(default="v1.0.0") # NEW: Track which model did the work

class RiskReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="joblisting.id")
    reason: str
    created_at: datetime = Field(default_factory=datetime.utcnow)