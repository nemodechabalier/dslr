import json
from typing import Dict

from data.models import DatasetStore


def train_one_vs_all(dataset_store: DatasetStore) -> Dict[str, list]:
    """Placeholder for one-vs-all logistic regression training.

    This function should train one classifier per house and return learned weights.
    """
    _ = dataset_store
    raise NotImplementedError("Logistic regression training is not implemented yet.")


def save_weights(weights: Dict[str, list], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(weights, file_obj, indent=2)
