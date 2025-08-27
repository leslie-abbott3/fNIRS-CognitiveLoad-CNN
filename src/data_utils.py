import numpy as np
import pandas as pd
import os, argparse

def generate_synthetic_data(samples=200, channels=16, length=200, save_dir="data/raw"):
    """
    Generate synthetic fNIRS-like data (sinusoidal + noise).
    Labels: 0=low load, 1=high load
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(samples):
        label = np.random.choice([0, 1])
        freq = 1 if label == 0 else 3
        data = np.array([
            np.sin(np.linspace(0, freq*np.pi, length)) + np.random.normal(0, 0.1, length)
            for _ in range(channels)
        ])
        filename = f"subject_{i}_{'low' if label==0 else 'high'}.csv"
        pd.DataFrame(data.T).to_csv(os.path.join(save_dir, filename), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()
    if args.generate:
        generate_synthetic_data(args.samples, args.channels, args.length)
        print("[INFO] Synthetic dataset generated in data/raw/")
