from pandas import DataFrame
import numpy as np
from data.models import DatasetStore
import json


def load_json(path: str):
    with open(path) as f:
        data = json.load(f)
    return data


def simozoide(z :float) -> float:
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    """Binary cross-entropy cost function"""
    h = simozoide(X @ theta)
    cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return cost