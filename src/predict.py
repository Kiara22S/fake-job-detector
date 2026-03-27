import joblib
import pandas as pd
from src.feature_engineering import extract_features, calculate_risk_score
from backend.risk_engine import RiskEngine
# 1. Load the Atomic Pipeline (Handles Vectorization + Model)
model = joblib.load("model/sentinel_pipeline.pkl")
risk_engine = RiskEngine()

def analyze_job(title, description, requirements, company_email=""):
    """
    Production-grade prediction function with Explanation Mapping.
    """
    # Create the raw DataFrame
    df = pd.DataFrame([{
        "title": title,
        "description": description,
        "requirements": requirements,
        "company_email": company_email,
        "company_profile": "",
        "location": "",
        "benefits": ""
    }])

    # 2. Feature Engineering (The 'Heuristics')
    df = extract_features(df)
    df["risk_score"] = df.apply(calculate_risk_score, axis=1)

    # 3. Text Pre-processing (Match the training input)
    df["full_text"] = (
        df["title"].fillna("") + " " + 
        df["description"].fillna("") + " " + 
        df["requirements"].fillna("")
    )

    # 4. ML Inference
    # The Pipeline handles the TF-IDF and structured features automatically
    probability = model.predict_proba(df)[0][1]
    
    # 5. EXPLANATION MAPPING (The 'Voice')
    # We pass the engineered row to the RiskEngine to get human-readable findings
    row_dict = df.iloc[0].to_dict()
    report = risk_engine.generate_report(row_dict, probability)

    return {
        "verdict": "Fraudulent" if probability > 0.5 else "Legitimate",
        "confidence": f"{round(probability * 100, 2)}%",
        "risk_details": report  # This contains your human-readable sentences
    }

if __name__ == "__main__":
    # Test Case
    test_title = "Data Entry Specialist"
    test_desc = "Work from home, earn $5000 weekly. No experience. Send $50 for kit."
    test_req = "Must have a laptop and bank account."
    
    result = analyze_job(test_title, test_desc, test_req, "hr-scams@gmail.com")
    
    import json
    print(json.dumps(result, indent=4))