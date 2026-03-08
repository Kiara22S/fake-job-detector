import pandas as pd
from feature_engineering import extract_features, calculate_risk_score


df = pd.read_csv("data/fake_job_postings.csv")

df = extract_features(df)

df["risk_score"] = df.apply(calculate_risk_score, axis=1)

print(df[["risk_score"]].head())