import pandas as pd
import pickle

from feature_engineering import extract_features, calculate_risk_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1 Load Dataset
# -----------------------------

df = pd.read_csv("data/fake_job_postings.csv")

print("Dataset Loaded")


# -----------------------------
# 2 Feature Engineering
# -----------------------------

df = extract_features(df)

# Apply risk scoring
df["risk_score"] = df.apply(calculate_risk_score, axis=1)

print("Risk Scores Generated")
print(df[["risk_score"]].head())


# -----------------------------
# 3 Define Features + Target
# -----------------------------

features = [
    "gmail_domain",
    "has_payment_request",
    "contains_urgent_words",
    "salary_mentioned",
    "location_missing",
    "description_length",
    "risk_score"
]

X = df[features]
y = df["fraudulent"]


# -----------------------------
# 4 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Data Split Complete")


# -----------------------------
# 5 Train Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Complete")


# -----------------------------
# 6 Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# 7 Save Model
# -----------------------------

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully.")