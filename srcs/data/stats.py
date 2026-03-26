import math
from typing import Dict, List, Tuple

from .models import FeatureStats
from .preprocess import is_nan


def var_skew_kurt(values: List[float], mean: float) -> Tuple[float, float, float]:
    variance_sum = 0.0
    m2_sum = 0.0
    m3_sum = 0.0
    m4_sum = 0.0

    for value in values:
        delta = value - mean
        delta2 = delta * delta
        variance_sum += delta2
        m2_sum += delta2
        m3_sum += delta2 * delta
        m4_sum += delta2 * delta2

    variance = variance_sum / len(values)
    if variance == 0:
        return 0.0, 0.0, 0.0

    m2 = m2_sum / len(values)
    m3 = m3_sum / len(values)
    m4 = m4_sum / len(values)

    skewness = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2 * m2)

    return variance, skewness, kurtosis


def _percentile_from_sorted(values: List[float], ratio: float) -> float:
    if not values:
        return float("nan")
    index = int(len(values) * ratio)
    if index >= len(values):
        index = len(values) - 1
    return values[index]


def compute_stats_for_matrix(feature_matrix: List[List[float]], feature_names: List[str]) -> FeatureStats:
    stats: FeatureStats = {}

    if not feature_matrix or not feature_names:
        return stats

    num_features = len(feature_names)

    for feature_idx in range(num_features):
        values = []
        missing = 0

        for row in feature_matrix:
            value = row[feature_idx]
            if isinstance(value, (int, float)) and not is_nan(value):
                values.append(value)
            else:
                missing += 1

        if not values:
            continue

        values.sort()
        mean = sum(values) / len(values)
        variance, skewness, kurtosis = var_skew_kurt(values, mean)

        feature_name = feature_names[feature_idx]
        stats[feature_name] = {
            "count": float(len(values)),
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "min": values[0],
            "25%": _percentile_from_sorted(values, 0.25),
            "50%": _percentile_from_sorted(values, 0.50),
            "75%": _percentile_from_sorted(values, 0.75),
            "max": values[-1],
            "range": values[-1] - values[0],
            "missing": float(missing),
        }

    return stats
