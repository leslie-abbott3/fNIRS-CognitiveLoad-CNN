import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def build_cnn(input_shape, conv_filters=[32,64], kernel_sizes=[5,3], dropout=0.3, dense_units=128):
    model = models.Sequential()
    model.add(layers.Conv1D(conv_filters[0], kernel_size=kernel_sizes[0], activation="relu", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(conv_filters[1], kernel_size=kernel_sizes[1], activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_baseline(model_type="svm"):
    if model_type == "svm":
        return SVC(kernel="rbf", probability=True)
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Unknown baseline type")
