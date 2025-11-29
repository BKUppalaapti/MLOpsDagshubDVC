import numpy as np
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import os

# --- Initialize DagsHub MLflow tracking ---
dagshub.init(
    repo_owner="krishnauppalapatiaws",
    repo_name="MLOpsDagshubDVC",
    mlflow=True
)

# --- Load train/test features locally ---
train_data = pd.read_csv("artifacts/features/tweet_emotions_train_features.csv")
test_data = pd.read_csv("artifacts/features/tweet_emotions_test_features.csv")

X_train = train_data.drop(columns=["sentiment"]).values
y_train = train_data["sentiment"].values

X_test = test_data.drop(columns=["sentiment"]).values
y_test = test_data["sentiment"].values

# --- Train model ---
clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# --- Evaluate on train set ---
y_train_pred = clf.predict(X_train)
y_train_proba = clf.predict_proba(X_train)[:, 1] if hasattr(clf, "predict_proba") else None

train_metrics = {
    "accuracy": accuracy_score(y_train, y_train_pred),
    "precision": precision_score(y_train, y_train_pred, average="weighted"),
    "recall": recall_score(y_train, y_train_pred, average="weighted"),
    "auc": roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else None
}

# --- Evaluate on test set ---
y_test_pred = clf.predict(X_test)
y_test_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

eval_metrics = {
    "accuracy": accuracy_score(y_test, y_test_pred),
    "precision": precision_score(y_test, y_test_pred, average="weighted"),
    "recall": recall_score(y_test, y_test_pred, average="weighted"),
    "auc": roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None
}

# --- Save metrics locally for DVC ---
os.makedirs("artifacts/metrics", exist_ok=True)
with open("artifacts/metrics/gb_train_metrics.json", "w") as f:
    json.dump(train_metrics, f)
with open("artifacts/metrics/gb_eval_metrics.json", "w") as f:
    json.dump(eval_metrics, f)

print("✅ Saved Gradient Boosting metrics locally for DVC tracking")

# --- Save model locally for DVC ---
os.makedirs("artifacts/models", exist_ok=True)
model_out = "artifacts/models/gb_model.pkl"
with open(model_out, "wb") as f:
    pickle.dump(clf, f)
print(f"✅ Gradient Boosting model saved locally at {model_out}")

# --- Log + Register in MLflow/DagsHub ---
with mlflow.start_run() as run:
    # Log metrics
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})

    # Log parameters
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("model_type", "GradientBoostingClassifier")

    # Log model artifact
    mlflow.sklearn.log_model(clf, "gb_model")

    # Register model in MLflow Model Registry
    model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri, "TweetSentimentModel")

    print(f"✅ Model registered as {result.name}, version {result.version}")