import csv
import json
import numpy as np

from typing import Dict, List

from pandas import DataFrame
from .utils import simozoide
from data.models import DatasetStore

"""
pour le one vs all
Gryffindor fliying, et notamant flying herbology
Slytherin divination, et notament divination avec astronomie
Hufflepuff pas de matières évidente par contre Herbology vs Astronomy, Herbologie vs Charms
Ravenclaw Muggle Studies et notament Muggle Studies 
"""

def normalized_value(dataset_store: DatasetStore, features: list[str], describe: Dict[str, Dict[str, float]]) -> DataFrame:
    selected_column = features + ["Hogwarts House"]
    data = dataset_store.clean_dataframe[selected_column].copy()
    data = data.dropna(subset=features)
    #print(data)
    for feature in features:
        mean = describe[feature]["mean"]
        std = describe[feature]["std"]
        data[feature] = (data[feature] - mean) / std
    #print(data)
    return data

def predict_houses(dataset_store: DatasetStore, data: Dict[str, list]) -> List[str]:
    """Placeholder for one-vs-all logistic regression prediction."""
    features = data["features"]
    weights = data["weights"]
    describe = data["describe"]
    normalized_data = normalized_value(dataset_store, features, describe)
    X = normalized_data[features].values
    predictions = []
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    for i in range(X_with_bias.shape[0]):
        x_i = X_with_bias[i]
        scores = {house: simozoide(np.dot(x_i, weights[house])) for house in houses}
        # print(f"Sample {i}, Scores: {scores}")  # Debug: print scores for each house
        predicted_house = max(scores, key=scores.get)
        predictions.append(predicted_house)
    write_predictions(predictions, "houses.csv")
    return predictions

def write_predictions(predictions: List[str], output_path: str = "houses.csv") -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in enumerate(predictions):
            writer.writerow([index, house])
