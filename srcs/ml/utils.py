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