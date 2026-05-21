import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data.io import load_dataset
from data.models import DatasetStore
from data.preprocess import (
    build_clean_dataframe,
    clean_nan,
    dataframe_to_feature_matrix,
    extract_feature_names,
    extract_houses,
)
from data.stats import compute_stats_for_matrix
from ml.utils import simozoide
from ml.train import gradiant_descent


DEFAULT_FEATURES = [
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Charms",
    "Flying",
]


def build_dataset_store(dataframe, feature_names):
    raw_features = dataframe_to_feature_matrix(dataframe, feature_names)
    clean_features = clean_nan(raw_features)
    stats_raw = compute_stats_for_matrix(raw_features, feature_names)
    stats_clean = compute_stats_for_matrix(clean_features, feature_names)
    clean_dataframe = build_clean_dataframe(dataframe, feature_names, clean_features)

    return DatasetStore(
        raw_dataframe=dataframe,
        clean_dataframe=clean_dataframe,
        feature_names=feature_names,
        houses=extract_houses(dataframe),
        raw_features=raw_features,
        clean_features=clean_features,
        stats_raw=stats_raw,
        stats_clean=stats_clean,
    )


def normalize_with_stats(dataframe, feature_names, stats):
    selected_columns = feature_names + ["Hogwarts House"]
    data = dataframe[selected_columns].copy()

    for feature in feature_names:
        mean = stats[feature]["mean"]
        std = stats[feature]["std"]
        data[feature] = data[feature].fillna(mean)
        data[feature] = (data[feature] - mean) / std

    return data


def train_one_vs_all(X, y, method):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    weights = {}

    m = len(y)
    X_with_bias = np.column_stack([np.ones(m), X])

    for house in houses:
        y_binary = (y == house).astype(float)
        theta = np.zeros(X_with_bias.shape[1])
        theta = gradiant_descent(X_with_bias, y_binary, theta, method)
        weights[house] = theta

    return weights


def predict_labels(dataframe, feature_names, weights, stats):
    normalized_data = normalize_with_stats(dataframe, feature_names, stats)
    X = normalized_data[feature_names].values
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

    predictions = []
    for i in range(X_with_bias.shape[0]):
        x_i = X_with_bias[i]
        scores = {house: simozoide(np.dot(x_i, weights[house])) for house in houses}
        predictions.append(max(scores, key=scores.get))

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one-vs-all logistic regression on a hold-out split.",
    )
    parser.add_argument(
        "datasets",
        nargs="?",
        default="datasets/dataset_train.csv",
        help="Path to CSV dataset file (example: datasets/dataset_train.csv)",
    )
    parser.add_argument(
        "method",
        nargs="?",
        default="batch",
        help="Gradient descent method (batch, stochastic, or mini-batch)",
    )
    parser.add_argument(
        "features",
        nargs="*",
        default=DEFAULT_FEATURES,
        help="Feature names to use for validation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the training dataset used for validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/validation split",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.datasets)
    if dataset is None:
        return 1

    feature_names = extract_feature_names(dataset)
    for feature in args.features:
        if feature not in feature_names:
            available = ", ".join(feature_names)
            print(
                f"Error: feature '{feature}' not found in dataset. Available features are: {available}"
            )
            return 2

    if args.method not in ["batch", "stochastic", "mini-batch"]:
        print("Error: invalid method. Choose 'batch', 'stochastic', or 'mini-batch'.")
        return 2

    train_df, val_df = train_test_split(
        dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset["Hogwarts House"],
    )

    train_store = build_dataset_store(train_df.reset_index(drop=True), args.features)
    val_df = val_df.reset_index(drop=True)

    X_train = train_store.clean_dataframe[args.features].values
    y_train = train_store.clean_dataframe["Hogwarts House"].values
    weights = train_one_vs_all(X_train, y_train, args.method)

    y_true = val_df["Hogwarts House"].values
    y_pred = predict_labels(val_df, args.features, weights, train_store.stats_clean)

    score = accuracy_score(y_true, y_pred)
    print(f"Validation accuracy: {score:.4f}")
    if score >= 0.98:
        print("Target reached: accuracy is at least 98%.")
    else:
        print("Target not reached yet: try different features, more iterations, or another split.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())