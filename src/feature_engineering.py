import pandas as pd
import re
import whois
from urllib.parse import urlparse
import datetime

def extract_features(df):

    df["description"] = df["description"].fillna("")
    df["company_profile"] = df["company_profile"].fillna("")
    df["location"] = df["location"].fillna("")

    # Feature 1
    df["description_length"] = df["description"].apply(lambda x: len(x.split()))

    # Feature 2
    df["has_payment_request"] = df["description"].str.contains(
        r"fee|payment|deposit|registration fee|processing fee",
        case=False
    ).astype(int)

    # Feature 3
    df["contains_urgent_words"] = df["description"].str.contains(
        r"urgent|immediate|hiring fast|limited slots",
        case=False
    ).astype(int)

    # Feature 4
    df["gmail_domain"] = df["company_profile"].str.contains(
        r"gmail.com|yahoo.com|outlook.com",
        case=False
    ).astype(int)

    # Feature 5
    df["salary_mentioned"] = df["description"].str.contains(
        r"\$|₹|salary|stipend",
        case=False
    ).astype(int)

    # Feature 6
    df["location_missing"] = df["location"].apply(lambda x: 1 if x == "" else 0)

    df["company_profile_length"] = df["company_profile"].apply(lambda x: len(x.split()))
    return df
# ---------- Risk Scoring ----------

def calculate_risk_score(row):

    score = 0

    score += 25 * row["has_payment_request"]
    score += 20 * row["gmail_domain"]
    score += 15 * row["contains_urgent_words"]
    score += 10 * row["salary_mentioned"]
    score += 10 * row["location_missing"]

    if row["description_length"] < 50:
        score += 5

    return score