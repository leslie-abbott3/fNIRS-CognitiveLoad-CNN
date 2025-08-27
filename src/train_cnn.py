import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os, argparse

def load_data(path):
    X, y = [], []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            arr = np.loadtxt(os.path.join(path, file), delimiter=",")
            X.append(arr)
            label = 0 if "low" in file else 1  # mock labels
            y.append(label)
    return np.array(X), np.array(y)

def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    X, y = load_data(args.data)
    X = np.expand_dims(X, -1)  # add channel dimension
    model = build_cnn(X.shape[1:])
    
    history = model.fit(X, y, epochs=args.epochs, batch_size=args.batch, validation_split=0.2)
    os.makedirs("models/saved_model", exist_ok=True)
    model.save("models/saved_model/cnn_model.h5")
    print("[INFO] Model saved to models/saved_model/cnn_model.h5")
