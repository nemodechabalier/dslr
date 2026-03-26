import csv
import json
from typing import Dict, List

from data.models import DatasetStore


def load_weights(weights_path: str) -> Dict[str, list]:
    with open(weights_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def predict_houses(dataset_store: DatasetStore, weights: Dict[str, list]) -> List[str]:
    """Placeholder for one-vs-all logistic regression prediction."""
    _ = dataset_store
    _ = weights
    raise NotImplementedError("Logistic regression prediction is not implemented yet.")


def write_predictions(predictions: List[str], output_path: str = "houses.csv") -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in enumerate(predictions):
            writer.writerow([index, house])
