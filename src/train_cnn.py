import os, argparse, yaml
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from models import build_cnn
from data_utils import generate_synthetic_data

def load_data(path):
    X, y = [], []
    for f in os.listdir(path):
        if f.endswith(".csv"):
            arr = np.loadtxt(os.path.join(path, f), delimiter=",")
            X.append(arr)
            label = 0 if "low" in f else 1
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X, y = load_data("data/processed")
    X = np.expand_dims(X, -1)

    model = build_cnn(X.shape[1:], **cfg["model"])
    logdir = "logs"
    os.makedirs(logdir, exist_ok=True)
    tb_callback = TensorBoard(log_dir=logdir)

    model.fit(X, y, batch_size=cfg["training"]["batch_size"],
              epochs=cfg["training"]["epochs"],
              validation_split=cfg["training"]["validation_split"],
              callbacks=[tb_callback])
    os.makedirs("models/saved_model", exist_ok=True)
    model.save("models/saved_model/cnn_model.h5")
    print("[INFO] Model saved.")

