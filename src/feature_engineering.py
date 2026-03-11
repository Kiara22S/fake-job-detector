import pandas as pd
import re
from datetime import datetime

# 1. FAANG-Level Weights Configuration (Total = 1.0)
# These weights ensure the final score is always between 0 and 100.
RISK_WEIGHTS = {
    "has_payment_request": 0.35,      # 35% - Highest risk factor
    "gmail_domain": 0.20,             # 20% - Use of free/non-corporate email
    "contains_urgent_words": 0.15,    # 15% - Social engineering/pressure
    "new_domain": 0.10,               # 10% - Suspicious or temporary URLs
    "salary_mentioned": 0.10,         # 10% - High-salary baiting
    "location_missing": 0.05,         # 5%  - Incomplete listing
    "short_description_bonus": 0.05   # 5%  - Low-effort post
}

def get_risk_category(score):
    """
    Categorizes the score into human-readable buckets.
    Used by the Frontend for color-coding and labels.
    """
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    else:
        return "High"

def calculate_risk_score(row):
    """
    Calculates a normalized risk score (0-100).
    Uses .get() for resilience against missing columns.
    """
    score = 0.0

    # Weighted contribution from binary features
    score += RISK_WEIGHTS["has_payment_request"] * row.get("has_payment_request", 0)
    score += RISK_WEIGHTS["gmail_domain"] * row.get("gmail_domain", 0)
    score += RISK_WEIGHTS["contains_urgent_words"] * row.get("contains_urgent_words", 0)
    score += RISK_WEIGHTS["new_domain"] * row.get("new_domain", 0)
    score += RISK_WEIGHTS["salary_mentioned"] * row.get("salary_mentioned", 0)
    score += RISK_WEIGHTS["location_missing"] * row.get("location_missing", 0)

    # Heuristic contribution (Short job descriptions)
    if row.get("description_length", 0) < 50:
        score += RISK_WEIGHTS["short_description_bonus"]

    # Multiply by 100 to get the final percentage scale
    return round(float(score) * 100, 2)

def extract_features(df):
    """
    Main Feature Engineering Pipeline.
    Processes raw text data into normalized risk metrics.
    """
    
    # Pre-processing: Standardize text and handle missing values
    df["description"] = df["description"].fillna("").astype(str)
    df["company_profile"] = df["company_profile"].fillna("").astype(str)
    df["location"] = df["location"].fillna("").astype(str)

    # --- Feature Extraction Logic ---

    # 1. Text Complexity
    df["description_length"] = df["description"].apply(lambda x: len(x.split()))

    # 2. Payment Patterns (Regex)
    payment_pattern = r"fee|payment|deposit|registration fee|processing fee"
    df["has_payment_request"] = df["description"].str.contains(payment_pattern, case=False).astype(int)

    # 3. Urgency Patterns
    urgent_pattern = r"urgent|immediate|hiring fast|limited slots"
    df["contains_urgent_words"] = df["description"].str.contains(urgent_pattern, case=False).astype(int)

    # 4. Email Domain Patterns
    email_pattern = r"gmail.com|yahoo.com|outlook.com|hotmail.com"
    df["gmail_domain"] = df["company_profile"].str.contains(email_pattern, case=False).astype(int)

    # 5. Financial Baiting Patterns
    salary_pattern = r"\$|₹|salary|stipend"
    df["salary_mentioned"] = df["description"].str.contains(salary_pattern, case=False).astype(int)

    # 6. Suspicious URL Patterns
    suspicious_pattern = r"bit.ly|tinyurl.com|wixsite.com|000webhost"
    df["new_domain"] = df["company_profile"].str.contains(suspicious_pattern, case=False).astype(int)

    # 7. Structural Completeness
    df["location_missing"] = df["location"].apply(lambda x: 1 if x.strip() == "" else 0)
    
    # Metadata for potential future use
    df["company_profile_length"] = df["company_profile"].apply(lambda x: len(x.split()))

    # --- FINAL SCORING & CATEGORIES ---
    # Apply the math logic to create the 0-100 score
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    
    # Map that score to a human-readable label
    df["risk_category"] = df["risk_score"].apply(get_risk_category)
    
    return df