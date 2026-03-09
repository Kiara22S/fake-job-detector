import pandas as pd
import re
import whois
from urllib.parse import urlparse
import datetime

# Moving weights here makes it easy to "tune" your model later
RISK_WEIGHTS = {
    "has_payment_request": 0.40,  # 40% of the total risk
    "gmail_domain": 0.20,
    "contains_urgent_words": 0.15,
    "salary_mentioned": 0.10,
    "location_missing": 0.10,
    "short_description_bonus": 0.05
}

def calculate_risk_score(row):
    """FAANG-level normalized scoring formula."""
    score = 0.0
    score += RISK_WEIGHTS["has_payment_request"] * row["has_payment_request"]
    score += RISK_WEIGHTS["gmail_domain"] * row["gmail_domain"]
    score += RISK_WEIGHTS["contains_urgent_words"] * row["contains_urgent_words"]
    score += RISK_WEIGHTS["salary_mentioned"] * row["salary_mentioned"]
    score += RISK_WEIGHTS["location_missing"] * row["location_missing"]

    if row["description_length"] < 50:
        score += RISK_WEIGHTS["short_description_bonus"]

    return round(float(score), 4)

def extract_features(df):
    # Data cleaning
    df["description"] = df["description"].fillna("")
    df["company_profile"] = df["company_profile"].fillna("")
    df["location"] = df["location"].fillna("")

    # Feature Extractions
    df["description_length"] = df["description"].apply(lambda x: len(x.split()))
    
    df["has_payment_request"] = df["description"].str.contains(
        r"fee|payment|deposit|registration fee|processing fee", case=False
    ).astype(int)

    df["contains_urgent_words"] = df["description"].str.contains(
        r"urgent|immediate|hiring fast|limited slots", case=False
    ).astype(int)

    df["gmail_domain"] = df["company_profile"].str.contains(
        r"gmail.com|yahoo.com|outlook.com", case=False
    ).astype(int)

    df["salary_mentioned"] = df["description"].str.contains(
        r"\$|₹|salary|stipend", case=False
    ).astype(int)

    df["new_domain"] = df["company_profile"].str.contains(
    r"bit.ly|tinyurl.com|wixsite.com|000webhost", 
    case=False, na=False
).astype(int)

    df["location_missing"] = df["location"].apply(lambda x: 1 if x == "" else 0)
    df["company_profile_length"] = df["company_profile"].apply(lambda x: len(x.split()))

    # CRITICAL: This is where you actually use the risk scoring function!
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    
    return df