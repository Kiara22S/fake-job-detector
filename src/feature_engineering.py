import pandas as pd
import re
from datetime import datetime

# 1. TUNED RISK WEIGHTS (Must sum to 1.0 for 0-100 normalization)
# We increased payment_request to 0.40 because it is the strongest fraud indicator.
RISK_WEIGHTS = {
    "has_payment_request": 0.40,      # 40% - Critical Flag
    "gmail_domain": 0.15,             # 15% - Use of free email providers
    "new_domain": 0.15,               # 15% - Suspicious/Temporary URLs
    "contains_urgent_words": 0.10,    # 10% - Psychological pressure
    "salary_mentioned": 0.10,         # 10% - High-salary baiting
    "location_missing": 0.05,         # 5%  - Incomplete professional profile
    "short_description_bonus": 0.05   # 5%  - Low-effort listing
}

def get_risk_category(score):
    """
    Maps the numerical score to a human-readable label.
    0–30: Low | 31–60: Medium | 61–100: High
    """
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    else:
        return "High"

def calculate_risk_score(row):
    """Calculates a normalized risk score from 0.0 to 100.0."""
    score = 0.0

    # Summing weighted contributions
    score += RISK_WEIGHTS["has_payment_request"] * row.get("has_payment_request", 0)
    score += RISK_WEIGHTS["gmail_domain"] * row.get("gmail_domain", 0)
    score += RISK_WEIGHTS["contains_urgent_words"] * row.get("contains_urgent_words", 0)
    score += RISK_WEIGHTS["new_domain"] * row.get("new_domain", 0)
    score += RISK_WEIGHTS["salary_mentioned"] * row.get("salary_mentioned", 0)
    score += RISK_WEIGHTS["location_missing"] * row.get("location_missing", 0)

    # Heuristic: Short descriptions add minor risk
    if row.get("description_length", 0) < 50:
        score += RISK_WEIGHTS["short_description_bonus"]

    # Convert to 0-100 scale and round
    return round(float(score) * 100, 2)

def extract_features(df):
    """
    Main Feature Engineering Pipeline.
    Processes raw text -> Extracts Features -> Calculates Score -> Assigns Category.
    """
    
    # Pre-processing: Standardize text and handle missing values
    df["description"] = df["description"].fillna("").astype(str)
    df["company_profile"] = df["company_profile"].fillna("").astype(str)
    df["location"] = df["location"].fillna("").astype(str)

    # --- Feature Extraction ---
    # 1. Text Complexity
    df["description_length"] = df["description"].apply(lambda x: len(x.split()))

    # 2. Payment/Fee Patterns (Regex)
    payment_pattern = r"fee|payment|deposit|registration fee|processing fee"
    df["has_payment_request"] = df["description"].str.contains(payment_pattern, case=False).astype(int)

    # 3. Urgency Patterns
    urgent_pattern = r"urgent|immediate|hiring fast|limited slots"
    df["contains_urgent_words"] = df["description"].str.contains(urgent_pattern, case=False).astype(int)

    # 4. Free Email Domain Patterns
    email_pattern = r"gmail.com|yahoo.com|outlook.com|hotmail.com"
    df["gmail_domain"] = df["company_profile"].str.contains(email_pattern, case=False).astype(int)

    # 5. Financial Baiting Patterns
    salary_pattern = r"\$|₹|salary|stipend"
    df["salary_mentioned"] = df["description"].str.contains(salary_pattern, case=False).astype(int)

    # 6. Suspicious URL/Domain Patterns
    suspicious_pattern = r"bit.ly|tinyurl.com|wixsite.com|000webhost"
    df["new_domain"] = df["company_profile"].str.contains(suspicious_pattern, case=False).astype(int)

    # 7. Structural Completeness
    df["location_missing"] = df["location"].apply(lambda x: 1 if x.strip() == "" else 0)

    # --- FINAL SCORING & LABELING ---
    # Generate the 0-100 numerical score
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    
    # Generate the Low/Medium/High label
    df["risk_category"] = df["risk_score"].apply(get_risk_category)
    
    return df