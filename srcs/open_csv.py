import pandas as pd
from pandas import DataFrame
import math

class InvalidValueError(ValueError):
    def __init__(self, msg="DataFrame is NULL"):
        super().__init__(msg)

def load(path: str) -> DataFrame:
    """
    Load a CSV file and display its dimensions.

    Args:
        path: Path to the CSV file

    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        dataset = pd.read_csv(path)

        print(f"Loading dataset of dimensions {dataset.shape}")

        return dataset

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: File '{path}' has invalid format.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def cal(dataset: DataFrame) -> dict :
    stats ={}
    features_columns = dataset.columns[6:]
    print(features_columns[1])
    i = 0
    for col in features_columns:
        feature = []
        for _, row in dataset.iterrows():
            if not math.isnan(row[col]):
                feature.append(row[col])
        stats[features_columns[i]] = {}
        stats[features_columns[i]]["count"] = len(feature)
        stats[features_columns[i]]["mean"] = sum(feature) / len(feature)
        stats[features_columns[i]]["std"] = 0
        feature.sort()
        stats[features_columns[i]]["min"] = feature[0]
        stats[features_columns[i]]["max"] = feature[len(feature) - 1]
        stats[features_columns[i]]["25%"] = feature[int(len(feature) * 0.25)]
        stats[features_columns[i]]["50%"] = feature[int(len(feature) * 0.5)]
        stats[features_columns[i]]["75%"] = feature[int(len(feature) * 0.75)]
        print(stats[features_columns[i]]["mean"])
        i += 1

    return stats


def take_grade(dataset: DataFrame) -> list :
    if dataset is None:
        raise InvalidValueError
    X = []
    features_columns = dataset.columns[6:]
    for _, row in dataset.iterrows():
        features = []
        for col in features_columns:
            features.append(row[col])
        X.append(features)
    return X

def take_house(dataset: DataFrame) -> list :
    Y = []
    for house in dataset["Hogwarts House"]:
        Y.append(house)
    return Y


def clean_grade(grade: list, house : list) -> list:
    indices_to_remove = []
    for idx, line in enumerate(grade):
        missing_count = 0
        for value in line:
            if isinstance(value, float) and math.isnan(value):
                missing_count += 1
        if missing_count >= 3:
            indices_to_remove.append(idx)

    for idx in sorted(indices_to_remove, reverse=True):
        del grade[idx]
        del house[idx]

    return grade, house
                
def calc_means(grade: list) -> list:
    means = [0.0] * len(grade[0])
    tots = [0] * len(grade[0])

    for line in grade:
        for idy, value in enumerate(line):
            if isinstance(value, (int, float)) and value == value:
                means[idy] += value
                tots[idy] += 1

    for i in range(len(means)):
        means[i] = means[i] / tots[i] if tots[i] > 0 else None
    print(means)
    return means

def clean_NaN(grade: list):
    if not grade:
        return grade

    means = calc_means(grade)
    for idx, line in enumerate(grade):
        for idy, value in enumerate(line):
            if isinstance(value, float) and math.isnan(value):
                if means[idy] is not None:
                    grade[idx][idy] = means[idy]
    return grade

    
def clean_NaN(grade: list):
    if not grade:
        return grade

    means = calc_means(grade)
    for idx, line in enumerate(grade):
        for idy, value in enumerate(line):
            if isinstance(value, float) and math.isnan(value):
                if means[idy] is not None:
                    grade[idx][idy] = means[idy]
    return grade
    
def take_clean_data()-> list:
    data = load("../dataset/dataset_train.csv")
    if data is None:
        data = load("./dataset/dataset_train.csv")

    if data is not None:
        for column in data.columns:
            size = data[column].notna().sum()
            print(f"{column}: {size}")
    cal(data)
    grade = take_grade(data)
    house = take_house(data)
    print(len(grade))
    print(len(house))
    clean_grade(grade,house)
    clean_NaN(grade)
    print(len(grade))
    print(len(house))
    # print(grade)
    return house, grade