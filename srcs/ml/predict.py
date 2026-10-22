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

def normalized_value(dataset_store: DatasetStore, features: list[str], stats: Dict[str, Dict[str, float]]) -> DataFrame:
    """Normalize a prediction dataset with the provided training statistics."""

    selected_column = features + ["Hogwarts House"]
    data = dataset_store.clean_dataframe[selected_column].copy()
    data = data.dropna(subset=features)
    for feature in features:
        mean = stats[feature]["mean"]
        std = stats[feature]["std"]
        data[feature] = (data[feature] - mean) / std
    return data


def predict_houses(dataset_store: DatasetStore, data: Dict[str, list]) -> List[str]:
    """Predict Hogwarts houses and write them to houses.csv."""

    features = data["features"]
    weights = data["weights"]
    stats = data.get("mean") and data.get("std")

    if isinstance(stats, dict):
        norm_stats = {feature: {"mean": data["mean"][feature], "std": data["std"][feature]} for feature in features}
    else:
        describe = data["describe"]
        norm_stats = {feature: {"mean": describe[feature]["mean"], "std": describe[feature]["std"]} for feature in features}

    normalized_data = normalized_value(dataset_store, features, norm_stats)
    X = normalized_data[features].values
    predictions = []
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    for i in range(X_with_bias.shape[0]):
        x_i = X_with_bias[i]
        scores = {house: simozoide(x_i @ weights[house]) for house in houses}
        print(f"Sample {i}, Scores: {scores}")  # Debug: print scores for each house
        predicted_house = max(scores, key=scores.get)
        predictions.append(predicted_house)
    write_predictions(predictions, "houses_" + data["method"] + ".csv")
    return predictions

def write_predictions(predictions: List[str], output_path: str = "houses.csv") -> None:
    """Write predicted houses to a CSV file with the required header."""

    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in enumerate(predictions):
            writer.writerow([index, house])
