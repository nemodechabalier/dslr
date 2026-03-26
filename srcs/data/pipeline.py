from typing import List, Optional

from .io import load_dataset
from .models import DatasetStore
from .preprocess import (
    build_clean_dataframe,
    clean_nan,
    dataframe_to_feature_matrix,
    extract_feature_names,
    extract_houses,
)
from .stats import compute_stats_for_matrix


def prepare_dataset(path: str) -> DatasetStore:
    dataset = load_dataset(path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to load dataset from '{path}'")

    feature_names = extract_feature_names(dataset)
    houses = extract_houses(dataset)

    raw_features = dataframe_to_feature_matrix(dataset, feature_names)
    clean_features = clean_nan(raw_features)

    stats_raw = compute_stats_for_matrix(raw_features, feature_names)
    stats_clean = compute_stats_for_matrix(clean_features, feature_names)

    clean_dataframe = build_clean_dataframe(dataset, feature_names, clean_features)

    return DatasetStore(
        raw_dataframe=dataset,
        clean_dataframe=clean_dataframe,
        feature_names=feature_names,
        houses=houses,
        raw_features=raw_features,
        clean_features=clean_features,
        stats_raw=stats_raw,
        stats_clean=stats_clean,
    )


def try_prepare_dataset(paths: List[str]) -> Optional[DatasetStore]:
    for path in paths:
        loaded = load_dataset(path)
        if loaded is None:
            continue

        feature_names = extract_feature_names(loaded)
        houses = extract_houses(loaded)

        raw_features = dataframe_to_feature_matrix(loaded, feature_names)
        clean_features = clean_nan(raw_features)

        stats_raw = compute_stats_for_matrix(raw_features, feature_names)
        stats_clean = compute_stats_for_matrix(clean_features, feature_names)
        clean_dataframe = build_clean_dataframe(loaded, feature_names, clean_features)

        return DatasetStore(
            raw_dataframe=loaded,
            clean_dataframe=clean_dataframe,
            feature_names=feature_names,
            houses=houses,
            raw_features=raw_features,
            clean_features=clean_features,
            stats_raw=stats_raw,
            stats_clean=stats_clean,
        )

    return None
