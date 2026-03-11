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

# Balance the dataset (Random Undersampling)
fraud_jobs = df[df['fraudulent'] == 1]
real_jobs = df[df['fraudulent'] == 0].sample(n=len(fraud_jobs) * 2, random_state=42) 
# We take 2x as many real jobs as fakes to keep it realistic but balanced.

df = pd.concat([fraud_jobs, real_jobs]).sample(frac=1, random_state=42) # Shuffle
# ---------------------------

print(f"Dataset Loaded and Balanced. Total rows: {len(df)}")


# -----------------------------
# 2 Feature Engineering
# -----------------------------
df = extract_features(df)

print("Features and Risk Metrics Generated")
print(df[["risk_score", "risk_category"]].head())

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
    max_features=5000,       # Increased from 2000
    ngram_range=(1, 2)
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
   n_estimators=500,        # Increased from 200
   max_depth=30,            # Allows the trees to be more detailed
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
# 9.1 Feature Importance Extraction
# -----------------------------

# 1. Get feature names from TF-IDF
tfidf_features = tfidf.get_feature_names_out()

# 2. Combine with your structured feature names
all_feature_names = list(tfidf_features) + features

# 3. Get importance values from the Random Forest
importances = model.feature_importances_

# 4. Create a DataFrame for easy sorting
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_df.head(15))

# Optional: Save this to a CSV for your project report
feature_importance_df.to_csv("model/feature_importance.csv", index=False)


# -----------------------------
# 10 Save Model
# -----------------------------

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# SAVE THIS: The Translator (TF-IDF)
with open("model/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\nModel and Vectorizer saved successfully.")