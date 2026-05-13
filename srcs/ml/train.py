import json
from typing import Dict
from pandas import DataFrame
import numpy as np
from .utils import simozoide, normalized_value

from data.models import DatasetStore

"""
pour le one vs all
Gryffindor fliying, et notamant flying herbology
Slytherin divination, et notament divination avec astronomie
Hufflepuff pas de matières évidente par contre Herbology vs Astronomy, Herbologie vs Charms
Ravenclaw Muggle Studies et notament Muggle Studies 
"""

def compute_cost(X, y, theta):
    """Binary cross-entropy cost function"""
    h = simozoide(X @ theta)
    cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return cost
    
def gradiant_descent(X, y, theta, alpha = 0.1, num_iterations = 1000):
    m = len(y)
    for iter in range(num_iterations):
        h = simozoide(X @ theta)
        gradient = (X.T @ (h - y)) / m
        theta = theta - alpha * gradient
        if iter % 100 == 0:
           cost = compute_cost(X, y, theta)
           print(f"Iteration {iter}, Cost: {cost}")
    return theta
        

def train_models(dataset_store: DatasetStore, features: list[str]) -> Dict[str, list]:
    if not isinstance(features, list) :
        raise ValueError("Pair plot requires at least 2 features.")

    for feature in features:
        if feature not in dataset_store.feature_names:
            available = ", ".join(dataset_store.feature_names)
            raise ValueError(
                f"Feature '{feature}' not found in dataset. "
                f"Available features are: {available}"
            )

    print(f"Features selected for train : {features}")
    normalized_data = normalized_value(dataset_store, features)
    X = normalized_data[features].values  # Convertir en NumPy array
    y = normalized_data["Hogwarts House"].values
    weights = train_one_vs_all(X, y)
    print(f"Trained weights: {weights}")
    save_json(features, weights, dataset_store.stats_clean, "datasets/logreg_weights.json")


def train_one_vs_all(X , y) -> Dict[str, list]:
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    weights = {}
    
    m = len(y)
    X_with_bias = np.column_stack([np.ones(m), X])
    for house in houses:
        y_binary = (y == house).astype(float)
        theta = np.zeros(X_with_bias.shape[1])
        theta = gradiant_descent(X_with_bias, y_binary, theta, 0.1, 10000)
        print(f"Trained theta for {house}: {theta}")
        weights[house] = theta.tolist()
    return weights


def save_json(features: list[str], weights: Dict[str, list], describe: Dict[str, float], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump({"features": features, "weights": weights, "describe": describe}, file_obj, indent=2)
