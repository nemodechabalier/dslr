import math
from typing import List, Optional

from pandas import DataFrame

from .exceptions import InvalidValueError


def is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def extract_feature_names(dataset: DataFrame, start_index: int = 6) -> List[str]:
    return list(dataset.columns[start_index:])


def extract_houses(dataset: DataFrame, house_column: str = "Hogwarts House") -> List[str]:
    return list(dataset[house_column])


def dataframe_to_feature_matrix(dataset: DataFrame, feature_names: List[str]) -> List[List[float]]:
    if dataset is None:
        raise InvalidValueError
    matrix = []
    for _, row in dataset.iterrows():
        matrix.append([row[col] for col in feature_names])
    return matrix


def compute_feature_means(feature_matrix: List[List[float]]) -> List[Optional[float]]:
    if not feature_matrix:
        return []

    means = [0.0] * len(feature_matrix[0])
    totals = [0] * len(feature_matrix[0])

    for line in feature_matrix:
        for idx, value in enumerate(line):
            if isinstance(value, (int, float)) and not is_nan(value):
                means[idx] += value
                totals[idx] += 1

    for idx in range(len(means)):
        means[idx] = means[idx] / totals[idx] if totals[idx] > 0 else None

    return means


def clean_nan(feature_matrix: List[List[float]]) -> List[List[float]]:
    if not feature_matrix:
        return feature_matrix

    means = compute_feature_means(feature_matrix)
    clean_matrix = [line[:] for line in feature_matrix]

    for row_idx, line in enumerate(clean_matrix):
        for col_idx, value in enumerate(line):
            if is_nan(value) and means[col_idx] is not None:
                clean_matrix[row_idx][col_idx] = means[col_idx]

    return clean_matrix


def build_clean_dataframe(
    raw_dataframe: DataFrame,
    feature_names: List[str],
    clean_feature_matrix: List[List[float]],
) -> DataFrame:
    clean_dataframe = raw_dataframe.copy()

    for row_idx, row in enumerate(clean_feature_matrix):
        for col_idx, value in enumerate(row):
            clean_dataframe.at[row_idx, feature_names[col_idx]] = value

    return clean_dataframe
