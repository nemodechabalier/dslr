import json
from typing import Dict
from pandas import DataFrame

from data.models import DatasetStore

def normalized_value(dataset_store: DatasetStore, features: list[str]) -> DataFrame:
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
    data = normalized_value(dataset_store, features)
    
    

    

    


def train_one_vs_all(dataset_store: DatasetStore) -> Dict[str, list]:
    """Placeholder for one-vs-all logistic regression training.

    This function should train one classifier per house and return learned weights.
    """
    _ = dataset_store


def save_weights(weights: Dict[str, list], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(weights, file_obj, indent=2)
