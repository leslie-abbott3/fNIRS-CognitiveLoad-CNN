import tensorflow as tf
import numpy as np, os, argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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
    parser.add_argument("--model", type=str, default="models/saved_model/cnn_model.h5")
    parser.add_argument("--data", type=str, default="data/processed")
    args = parser.parse_args()

    X, y = load_data(args.data)
    X = np.expand_dims(X, -1)

    model = tf.keras.models.load_model(args.model)
    preds = (model.predict(X) > 0.5).astype("int32")
    print("\nClassification Report:\n", classification_report(y, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y, preds))
    print("\nROC-AUC:", roc_auc_score(y, preds))

