from pandas import DataFrame
import numpy as np
from data.models import DatasetStore
import json


def load_json(path: str):
    """Load and return JSON data from a file path."""

    with open(path) as f:
        data = json.load(f)
    return data


def simozoide(z :float) -> float:
    """Compute the sigmoid activation for z."""

    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    """Compute the binary cross-entropy cost for logistic regression."""

    h = simozoide(X @ theta)
    cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return cost