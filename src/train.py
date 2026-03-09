import pandas as pd
import pickle

from feature_engineering import extract_features, calculate_risk_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack


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
# 3 Combine Text Fields
# -----------------------------

df["full_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("")
)


# -----------------------------
# 4 TF-IDF Text Features
# -----------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=2000
)

X_text = tfidf.fit_transform(df["full_text"])


# -----------------------------
# 5 Structured Features
# -----------------------------

features = [
    "gmail_domain",
    "has_payment_request",
    "contains_urgent_words",
    "salary_mentioned",
    "location_missing",
    "description_length",
    "risk_score",
    "new_domain"
]

X_structured = df[features].values


# -----------------------------
# 6 Combine Features
# -----------------------------

X = hstack([X_text, X_structured])
y = df["fraudulent"]

print("Text feature shape:", X_text.shape)
print("Structured feature shape:", X_structured.shape)
print("Final feature shape:", X.shape)


# -----------------------------
# 7 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Data Split Complete")


# -----------------------------
# 8 Train Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Complete")


# -----------------------------
# 9 Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# 10 Save Model
# -----------------------------

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# SAVE THIS: The Translator (TF-IDF)
with open("model/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\nModel and Vectorizer saved successfully.")