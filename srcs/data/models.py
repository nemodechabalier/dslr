from dataclasses import dataclass, field
from typing import Dict, List

from pandas import DataFrame


FeatureStats = Dict[str, Dict[str, float]]


@dataclass
class DatasetStore:
    raw_dataframe: DataFrame
    clean_dataframe: DataFrame
    feature_names: List[str]
    houses: List[str]
    raw_features: List[List[float]]
    clean_features: List[List[float]]
    stats_raw: FeatureStats = field(default_factory=dict)
    stats_clean: FeatureStats = field(default_factory=dict)
