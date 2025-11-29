import os
import pandas as pd
import re

# --- Text cleaning ---
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)   # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

# --- Preprocess dataframe ---
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if "content" in df.columns:
        df["content"] = df["content"].apply(clean_text)
    df["sentiment"] = df["sentiment"].astype(int)
    return df

# --- Save processed splits locally (for DVC tracking) ---
def save_processed_local(df: pd.DataFrame, filename: str) -> None:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs("artifacts/processed", exist_ok=True)
    save_path = f"artifacts/processed/{base_name}.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ Saved processed file locally at {save_path}")

# --- Main ---
def main():
    raw_dir = "artifacts/raw"
    if not os.path.exists(raw_dir):
        print("No raw splits found locally. Run ingestion first.")
        return

    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not files:
        print("No raw CSV files found in artifacts/raw.")
        return

    for file in files:
        file_path = os.path.join(raw_dir, file)
        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path)
        processed_df = preprocess_df(df)
        save_processed_local(processed_df, filename=file)

if __name__ == "__main__":
    main()