import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os, argparse

def preprocess(input_path, output_path):
    """
    Preprocess fNIRS dataset:
    - Load CSV files
    - Normalize signals
    - Save processed data
    """
    all_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    os.makedirs(output_path, exist_ok=True)

    for file in all_files:
        df = pd.read_csv(os.path.join(input_path, file))
        X = df.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pd.DataFrame(X_scaled).to_csv(os.path.join(output_path, file), index=False)
        print(f"[INFO] Processed {file} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()
    preprocess(args.input, args.output)
