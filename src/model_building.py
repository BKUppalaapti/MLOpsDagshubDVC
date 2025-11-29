import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn

# --- Train model ---
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns=["sentiment"])
    y_train = train_df["sentiment"]

    X_test = test_df.drop(columns=["sentiment"])
    y_test = test_df["sentiment"]

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "auc": roc_auc_score(y_train, y_train_prob)
    }

    eval_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "auc": roc_auc_score(y_test, y_test_prob)
    }

    return model, train_metrics, eval_metrics

# --- Save model locally ---
def save_model_local(model):
    os.makedirs("artifacts/models", exist_ok=True)
    model_path = "artifacts/models/logreg_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved Logistic Regression model locally at {model_path}")
    return model_path

# --- Save metrics locally ---
def save_metrics(train_metrics, eval_metrics):
    os.makedirs("artifacts/metrics", exist_ok=True)
    train_path = "artifacts/metrics/logreg_train_metrics.json"
    eval_path = "artifacts/metrics/logreg_eval_metrics.json"

    pd.DataFrame([train_metrics]).to_json(train_path, orient="records", lines=True)
    pd.DataFrame([eval_metrics]).to_json(eval_path, orient="records", lines=True)

    print(f"✅ Saved train metrics at {train_path}")
    print(f"✅ Saved eval metrics at {eval_path}")

    return train_path, eval_path

# --- Log to MLflow ---
def log_to_mlflow(model, train_metrics, eval_metrics):
    mlflow.set_experiment("tweet_emotions_experiment")
    with mlflow.start_run():
        mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 500})
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})
        mlflow.sklearn.log_model(model, "logreg_model")

# --- Main ---
def main():
    features_dir = "artifacts/features"
    if not os.path.exists(features_dir):
        print("No feature files found locally. Run feature_engineering first.")
        return

    train_df = pd.read_csv(os.path.join(features_dir, "tweet_emotions_train_features.csv"))
    test_df = pd.read_csv(os.path.join(features_dir, "tweet_emotions_test_features.csv"))

    model, train_metrics, eval_metrics = train_model(train_df, test_df)

    # Save model locally
    save_model_local(model)

    # Save metrics locally
    save_metrics(train_metrics, eval_metrics)

    # Log to MLflow
    log_to_mlflow(model, train_metrics, eval_metrics)

if __name__ == "__main__":
    main()