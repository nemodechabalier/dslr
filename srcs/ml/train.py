import json
from typing import Dict
from matplotlib.pylab import shuffle
from pandas import DataFrame
import numpy as np
from .predict import write_predictions
from .utils import simozoide, compute_cost

from data.models import DatasetStore

"""
pour le one vs all
Gryffindor fliying, et notamant flying herbology
Slytherin divination, et notament divination avec astronomie
Hufflepuff pas de matières évidente par contre Herbology vs Astronomy, Herbologie vs Charms
Ravenclaw Muggle Studies et notament Muggle Studies 
"""

def normalized_value(dataset_store: DatasetStore, features: list[str]) -> DataFrame:
    """Return a normalized copy of the selected feature columns."""

    selected_column = features + ["Hogwarts House"]
    data = dataset_store.clean_dataframe[selected_column].copy()
    data = data.dropna(subset=features)
    #print(data)
    for feature in features:
        mean = dataset_store.stats_clean[feature]["mean"]
        std = dataset_store.stats_clean[feature]["std"]
        data[feature] = (data[feature] - mean) / std
    #print(data)
    return data


def gradiant_descent(X, y, theta, method = 'batch'):
    """Optimize theta with batch, stochastic, or mini-batch gradient descent."""

    if method == 'batch':
        alpha = 0.1
        num_iterations = 10000
        m = len(y)
        for epoch in range(num_iterations):
            h = simozoide(X @ theta)
            gradient = (X.T @ (h - y)) / m
            theta = theta - alpha * gradient
    if method == 'stochastic':
        alpha = 0.1
        num_epochs = 100
        m = len(y)
        for epoch in range(num_epochs):
            indices = np.random.permutation(m)
            for i in indices:
                x_i = X[i]
                y_i = y[i]
                h_i = simozoide(np.dot(x_i, theta))
                gradient = (h_i - y_i) * x_i
                theta = theta - alpha * gradient
    if method == 'mini-batch':
        alpha = 0.1
        num_epochs = 100
        batch_size = 32
        m = len(y)
        for epoch in range(num_epochs):
            for start_idx in range(0, m, batch_size):
                end_idx = min(start_idx + batch_size, m)
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                h_batch = simozoide(X_batch @ theta)
                gradient = (X_batch.T @ (h_batch - y_batch)) / len(y_batch)
                theta = theta - alpha * gradient
    return theta


def train_models(dataset_store: DatasetStore, features: list[str], method: str = 'batch') -> Dict[str, list]:
    """Train the one-vs-all logistic regression models and save their weights."""

    if not isinstance(features, list) :
        raise ValueError("Pair plot requires at least 2 features.")

    for feature in features:
        if feature not in dataset_store.feature_names:
            available = ", ".join(dataset_store.feature_names)
            raise ValueError(
                f"Feature '{feature}' not found in dataset. "
                f"Available features are: {available}"
            )

    if method not in ['batch', 'stochastic', 'mini-batch']:
        raise ValueError("Invalid method. Choose 'batch', 'stochastic', or 'mini-batch'.")

    print(f"Features selected for train : {features}")
    normalized_data = normalized_value(dataset_store, features)
    X = normalized_data[features].values  # Convertir en NumPy array
    y = normalized_data["Hogwarts House"].values
    #write_predictions(y, "true_houses.csv")
    weights = train_one_vs_all(X, y, method)
    print(f"Trained weights: {weights}")
    save_json(features, weights, dataset_store.stats_clean, "datasets/logreg_weights.json")


def train_one_vs_all(X , y, method) -> Dict[str, list]:
    """Train one binary classifier per Hogwarts house."""

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    weights = {}
    
    m = len(y)
    X_with_bias = np.column_stack([np.ones(m), X])
    for house in houses:
        y_binary = (y == house).astype(float)
        theta = np.zeros(X_with_bias.shape[1])
        theta = gradiant_descent(X_with_bias, y_binary, theta, method)
        print(f"Trained theta for {house}: {theta}")
        weights[house] = theta.tolist()
    return weights


def save_json(features: list[str], weights: Dict[str, list], describe: Dict[str, float], output_path: str) -> None:
    """Serialize the trained model metadata and weights to JSON."""

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump({"features": features, "weights": weights, "describe": describe}, file_obj, indent=2)
