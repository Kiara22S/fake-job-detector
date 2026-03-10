from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

# Table 1: Users
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    email: str = Field(unique=True)
    password: str

# Table 2: JobListing (UPDATED for AI Integration)
class JobListing(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    company: str
    description: str
    company_url: str = Field(default="https://unknown.com")
    
    # --- NEW FIELDS FOR KIARA'S ML MODEL ---
    location: Optional[str] = Field(default="")
    company_profile: Optional[str] = Field(default="")
    requirements: Optional[str] = Field(default="")
    # ---------------------------------------

    ml_risk_score: float = 0.0
    is_fake: bool = False

# Table 3: RiskReports
class RiskReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="joblisting.id")
    reason: str
    created_at: datetime = Field(default_factory=datetime.utcnow)