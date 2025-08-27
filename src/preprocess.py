import os, argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    return butter(order, [low, high], btype="band")

def preprocess_file(filepath, outdir, fs=10):
    df = pd.read_csv(filepath)
    X = df.values.T
    # bandpass filter (0.01 - 0.1 Hz typical for fNIRS)
    b, a = butter_bandpass(0.01, 0.1, fs)
    X_filtered = filtfilt(b, a, X)
    # normalize
    X_scaled = StandardScaler().fit_transform(X_filtered.T)
    outpath = os.path.join(outdir, os.path.basename(filepath))
    pd.DataFrame(X_scaled).to_csv(outpath, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for f in os.listdir(args.input):
        if f.endswith(".csv"):
            preprocess_file(os.path.join(args.input, f), args.output)
            print(f"[INFO] Processed {f}")

