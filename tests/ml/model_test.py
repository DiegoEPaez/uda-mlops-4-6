from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import ClassifierMixin

from src.ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def train_data():
    X_train = np.array([[1, 0.5], [2, 0.4], [3, 0.6]])
    y_train = np.array([0, 1, 0])

    return X_train, y_train


def test_train(train_data):
    X_train, y_train = train_data

    model = train_model(X_train, y_train)

    assert isinstance(model, ClassifierMixin)


def test_inference(train_data):
    X_train, y_train = train_data
    rf = train_model(X_train, y_train)
    pred = inference(rf, X_train)

    assert isinstance(pred, np.ndarray)


def test_compute_model_metrics(monkeypatch):
    mock_precision = Mock(return_value=42)
    mock_recall = Mock(return_value=42)

    monkeypatch.setattr("src.ml.model.precision_score", mock_precision)
    monkeypatch.setattr("src.ml.model.recall_score", mock_recall)

    compute_model_metrics(np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    assert mock_precision.called
    assert mock_recall.called
