import joblib
import pandas as pd
from feature_engineering import extract_features, calculate_risk_score

model = joblib.load("model/model.pkl")

def analyze_job(description, company_email=""):

    df = pd.DataFrame([{
        "description": description,
        "company_email": company_email,
        "company_profile": "",
        "location": "",
        "title": "",
        "requirements": "",
        "benefits": ""
    }])

    df = extract_features(df)

    df["risk_score"] = df.apply(calculate_risk_score, axis=1)

    # Get features used during training
    features = model.feature_names_in_

    # Ensure all required columns exist
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "risk_score": int(df["risk_score"].iloc[0])
    }


if __name__ == "__main__":

    test_description = """
    Work from home and earn $5000 weekly.
    No experience required.
    Immediate hiring.
    """

    result = analyze_job(test_description, "hr@gmail.com")

    print("Prediction Result:")
    print(result)