import numpy as np
import pandas as pd
import os
import boto3
import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- List all files in sourcedata folder ---
def list_s3_files(bucket: str, prefix: str = "sourcedata/") -> list:
    try:
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" not in response:
            print(f"No files found under {prefix} in bucket {bucket}")
            return []
        files = [f"s3://{bucket}/{obj['Key']}" for obj in response["Contents"] if obj["Key"].endswith(".csv")]
        return files
    except Exception as e:
        print(f"Error: Unable to list files in S3 bucket {bucket}/{prefix}")
        print(e)
        raise

# --- Load data from S3 file ---
def load_data(data_url: str) -> pd.DataFrame:
    try:
        # Requires: pip install s3fs fsspec
        df = pd.read_csv(data_url, storage_options={"anon": False})
        return df
    except Exception as e:
        print(f"Error loading {data_url}")
        print(e)
        raise

# --- Preprocess ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'tweet_id' in df.columns:
            df = df.drop(columns=['tweet_id'])
        if 'content' in df.columns:
            df['content'] = df['content'].apply(lambda x: html.unescape(str(x)))
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        le = LabelEncoder()
        final_df['sentiment'] = le.fit_transform(final_df['sentiment'])
        return final_df
    except Exception as e:
        print(f"Error preprocessing data")
        print(e)
        raise

# --- Scale numeric features only ---
def scale_features(train_data: pd.DataFrame, test_data: pd.DataFrame):
    try:
        y_train = train_data['sentiment']
        y_test = test_data['sentiment']
        text_train = train_data['content']
        text_test = test_data['content']

        numeric_cols = train_data.drop(columns=['sentiment', 'content']).columns

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_data[numeric_cols])
            X_test_scaled = scaler.transform(test_data[numeric_cols])
            train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols)
            test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols)
        else:
            train_scaled = pd.DataFrame()
            test_scaled = pd.DataFrame()

        train_scaled['content'] = text_train.values
        train_scaled['sentiment'] = y_train.values
        test_scaled['content'] = text_test.values
        test_scaled['sentiment'] = y_test.values

        return train_scaled, test_scaled
    except Exception as e:
        print(f"Error scaling features")
        print(e)
        raise

# --- Save train/test splits locally (for DVC tracking) ---
def save_data_local(train_data: pd.DataFrame, test_data: pd.DataFrame, filename: str) -> None:
    try:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        os.makedirs("artifacts/raw", exist_ok=True)
        train_path = f"artifacts/raw/{base_name}_train.csv"
        test_path = f"artifacts/raw/{base_name}_test.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(f"✅ Saved train/test splits locally under artifacts/raw for {filename}")
    except Exception as e:
        print(f"Error saving data for {filename}")
        print(e)
        raise

# --- Main ---
def main():
    bucket = "mlops-dagshub-dvc-mlflow-experimenttracking"
    files = list_s3_files(bucket=bucket, prefix="sourcedata/")

    if not files:
        print("No source files found in S3.")
        return

    for file in files:
        try:
            print(f"Processing {file}...")
            df = load_data(file)
            final_df = preprocess_data(df)
            train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
            train_scaled, test_scaled = scale_features(train_data, test_data)
            save_data_local(train_scaled, test_scaled, filename=file)
        except Exception as e:
            print(f"Failed to process {file}: {e}")

if __name__ == "__main__":
    main()