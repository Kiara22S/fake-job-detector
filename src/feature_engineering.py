import pandas as pd
import re
from datetime import datetime

# 1. FAANG-Level Weights Configuration (Total = 1.0)
RISK_WEIGHTS = {
    "has_payment_request": 0.35,      # 35% - Highest risk
    "gmail_domain": 0.20,             # 20% - Free email usage
    "contains_urgent_words": 0.15,    # 15% - Social engineering
    "new_domain": 0.10,               # 10% - Suspicious/Free URLs
    "salary_mentioned": 0.10,         # 10% - Scam bait
    "location_missing": 0.05,         # 5%  - Poor listing structure
    "short_description_bonus": 0.05   # 5%  - Low effort post
}

def get_risk_category(score):
    """
    Categorizes the score into human-readable buckets.
    0–30: Low | 31–60: Medium | 61–100: High
    """
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    else:
        return "High"

def calculate_risk_score(row):
    """Calculates a normalized risk score (0-100)."""
    score = 0.0

    # Binary feature contribution
    score += RISK_WEIGHTS["has_payment_request"] * row.get("has_payment_request", 0)
    score += RISK_WEIGHTS["gmail_domain"] * row.get("gmail_domain", 0)
    score += RISK_WEIGHTS["contains_urgent_words"] * row.get("contains_urgent_words", 0)
    score += RISK_WEIGHTS["new_domain"] * row.get("new_domain", 0)
    score += RISK_WEIGHTS["salary_mentioned"] * row.get("salary_mentioned", 0)
    score += RISK_WEIGHTS["location_missing"] * row.get("location_missing", 0)

    # Heuristic contribution
    if row.get("description_length", 0) < 50:
        score += RISK_WEIGHTS["short_description_bonus"]

    return round(float(score) * 100, 2)

def extract_features(df):
    """Production-grade pipeline for feature extraction and scoring."""
    
    # Pre-processing
    df["description"] = df["description"].fillna("").astype(str)
    df["company_profile"] = df["company_profile"].fillna("").astype(str)
    df["location"] = df["location"].fillna("").astype(str)

    # --- Feature Extraction ---
    df["description_length"] = df["description"].apply(lambda x: len(x.split()))
    
    # Patterns
    payment_pattern = r"fee|payment|deposit|registration fee|processing fee"
    df["has_payment_request"] = df["description"].str.contains(payment_pattern, case=False).astype(int)

    urgent_pattern = r"urgent|immediate|hiring fast|limited slots"
    df["contains_urgent_words"] = df["description"].str.contains(urgent_pattern, case=False).astype(int)

    email_pattern = r"gmail.com|yahoo.com|outlook.com|hotmail.com"
    df["gmail_domain"] = df["company_profile"].str.contains(email_pattern, case=False).astype(int)

    salary_pattern = r"\$|₹|salary|stipend"
    df["salary_mentioned"] = df["description"].str.contains(salary_pattern, case=False).astype(int)

    suspicious_pattern = r"bit.ly|tinyurl.com|wixsite.com|000webhost"
    df["new_domain"] = df["company_profile"].str.contains(suspicious_pattern, case=False).astype(int)

    df["location_missing"] = df["location"].apply(lambda x: 1 if x.strip() == "" else 0)
    df["company_profile_length"] = df["company_profile"].apply(lambda x: len(x.split()))

    # --- THE FINAL SCORING & CATEGORIES ---
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    df["risk_category"] = df["risk_score"].apply(get_risk_category)
    
    return df