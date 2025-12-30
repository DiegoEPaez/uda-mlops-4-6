import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data

BASE = "model"
PATH_MODEL = os.path.join(BASE, "rf.joblib")
PATH_ENCODER = os.path.join(BASE, "encoder.joblib")
PATH_LB = os.path.join(BASE, "lb.joblib")


def save_model(model, encoder, lb):
    joblib.dump(model, PATH_MODEL)
    joblib.dump(encoder, PATH_ENCODER)
    joblib.dump(lb, PATH_LB)


def load_model():
    return joblib.load(PATH_MODEL), joblib.load(PATH_ENCODER), joblib.load(PATH_LB)


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    return rf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_metrics_slices(data, model, cat_features, encoder, lb):
    X, y, _, _ = process_data(
        data, cat_features, "salary", training=False, encoder=encoder, lb=lb
    )
    metrics = [[None, None, *compute_model_metrics(y, model.predict(X))]]

    for var in cat_features:
        values = data[var].unique()
        for val in values:
            slice = data[data[var] == val]

            X, y, _, _ = process_data(
                slice, cat_features, "salary", training=False, encoder=encoder, lb=lb
            )
            y_preds = model.predict(X)
            metrics_per_slice = compute_model_metrics(y, y_preds)
            metrics.append([var, val] + list(metrics_per_slice))

    return pd.DataFrame(
        metrics, columns=["feature", "slice", "precision", "recall", "fbeta"]
    )


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict_proba(X)[:, 1]
