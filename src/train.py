import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import your custom logic (Ensure these functions exist in feature_engineering.py)
from feature_engineering import extract_features, calculate_risk_score

# -----------------------------
# 1. Dataset Preparation
# -----------------------------
df = pd.read_csv("data/fake_job_postings.csv")

# Strategic Undersampling to handle class imbalance (Enterprise standard)
fraud_jobs = df[df['fraudulent'] == 1]
real_jobs = df[df['fraudulent'] == 0].sample(n=len(fraud_jobs) * 2, random_state=42)
df = pd.concat([fraud_jobs, real_jobs]).sample(frac=1, random_state=42)

# -----------------------------
# 2. Engineering & Feature Selection
# -----------------------------
df = extract_features(df)
df["risk_score"] = df.apply(calculate_risk_score, axis=1)

# Combine text fields for NLP processing
df["full_text"] = (
    df["title"].fillna("") + " " + 
    df["description"].fillna("") + " " + 
    df["requirements"].fillna("")
)

# Define feature groups
text_feature = "full_text"
structured_features = [
    "gmail_domain", "has_payment_request", "contains_urgent_words",
    "salary_mentioned", "location_missing", "description_length",
    "risk_score", "new_domain"
]

X = df[[text_feature] + structured_features]
y = df["fraudulent"]

# -----------------------------
# 3. The Atomic Pipeline Architecture
# -----------------------------
# This block ensures TF-IDF and structured data are handled simultaneously
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2)), text_feature),
        ('struct', 'passthrough', structured_features)
    ]
)

# The Full Pipeline: Preprocess -> Classify
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        class_weight={0: 1, 1: 8},  # 8x penalty for missing a scam (High Recall)
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# 4. Training & Evaluation
# -----------------------------
# 'stratify' ensures training/test sets have the same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Sentinel Production-Grade Pipeline...")
pipeline.fit(X_train, y_train)

# Metrics
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*40)
print(f"MODEL ACCURACY: {accuracy:.2%}")
print("="*40)

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions, target_names=['Legitimate', 'Fraudulent']))

# -----------------------------
# 5. Feature Importance Extraction (The Proof)
# -----------------------------
tfidf_vocab = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
all_features = list(tfidf_vocab) + structured_features
importances = pipeline.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTOP 15 FRAUD INDICATORS:")
print(importance_df.head(15))

# -----------------------------
# 6. Exporting the Artifact
# -----------------------------
joblib.dump(pipeline, "model/sentinel_pipeline.pkl")
importance_df.to_csv("model/feature_importance.csv", index=False)
print("\nSuccess: Production artifact saved at 'model/sentinel_pipeline.pkl'")