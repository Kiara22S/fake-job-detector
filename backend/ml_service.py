import joblib
import os
import pandas as pd
from scipy.sparse import hstack
import logging

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self, model_path="model/model.pkl", tfidf_path="model/tfidf.pkl"):
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None
        self.tfidf = joblib.load(tfidf_path) if os.path.exists(tfidf_path) else None
        
        if not self.model or not self.tfidf:
            logger.error("ML Service failed to load assets. Predictions will be 'Unverified'.")

    def predict(self, job_data, processed_df):
        if not self.model or not self.tfidf:
            return "Unverified", 0.0

        # Prepare Text Data
        full_text = f"{job_data.title} {job_data.description} {getattr(job_data, 'requirements', '')}"
        X_text = self.tfidf.transform([full_text])

        # Prepare Structured Features
        features_list = [
            "gmail_domain", "has_payment_request", "contains_urgent_words",
            "salary_mentioned", "location_missing", "description_length",
            "risk_score", "new_domain"
        ]
        X_structured = processed_df[features_list].values
        
        # Combine and Predict
        X_final = hstack([X_text, X_structured])
        prediction_bin = self.model.predict(X_final)[0]
        
        # Integrate Model Probability (Task 3 Logic)
        # We fetch the probability of being "Fake" (class 1)
        prob = self.model.predict_proba(X_final)[0][1] 
        
        verdict = "Fake" if prediction_bin == 1 else "Real"
        return verdict, float(prob)

# Singleton instance
ml_engine = MLService()