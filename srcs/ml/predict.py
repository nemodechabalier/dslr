import csv
import json
from typing import Dict, List
from .utils import simozoide, normalized_value


from data.models import DatasetStore

"""
pour le one vs all
Gryffindor fliying, et notamant flying herbology
Slytherin divination, et notament divination avec astronomie
Hufflepuff pas de matières évidente par contre Herbology vs Astronomy, Herbologie vs Charms
Ravenclaw Muggle Studies et notament Muggle Studies 
"""


def predict_houses(dataset_store: DatasetStore, data: Dict[str, list]) -> List[str]:
    """Placeholder for one-vs-all logistic regression prediction."""
    features = data["features"]
    weights = data["weights"]
    describe = data["describe"]
    normalized_data = normalized_value(dataset_store, features)
    print(data)
    X = normalized_data[features].values  # Convertir en NumPy array
    y = normalized_data["Hogwarts House"].values

    
    
    


def write_predictions(predictions: List[str], output_path: str = "houses.csv") -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in enumerate(predictions):
            writer.writerow([index, house])
