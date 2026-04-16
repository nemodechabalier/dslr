from data.models import DatasetStore
import matplotlib.pyplot as plt
import math

def is_nan(value) -> bool:
    return isinstance(value, (float, int)) and math.isnan(value)

def run_scatter_plot(dataset_store: DatasetStore, feature_1: str, feature_2: str) -> None:
    """Placeholder for scatter plot visualization.

    Expected question:
    What are the two features that are similar?
    """
    if feature_1 not in dataset_store.feature_names or feature_2 not in dataset_store.feature_names:
        available = ", ".join(dataset_store.feature_names)
        raise ValueError(
            f"Feature '{feature_1}' not found in dataset. "
            f"Available features are: {available}"
        )

    idx_1 = dataset_store.feature_names.index(feature_1)
    idx_2 = dataset_store.feature_names.index(feature_2)
    

    by_house = {
        "Gryffindor": {"x": [], "y": []},
        "Hufflepuff": {"x": [], "y": []},
        "Ravenclaw": {"x": [], "y": []},
        "Slytherin": {"x": [], "y": []},
    }

    for house, row in zip(dataset_store.houses, dataset_store.raw_features):
        if house not in by_house:
            continue

        x = row[idx_1]
        y = row[idx_2]
        if is_nan(x) or is_nan(y):
            continue
        by_house[house]["x"].append(x)
        by_house[house]["y"].append(y)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(by_house["Gryffindor"]["x"], by_house["Gryffindor"]["y"], alpha=0.6, label="Gryffindor", color="red")
    plt.scatter(by_house["Hufflepuff"]["x"], by_house["Hufflepuff"]["y"], alpha=0.6, label="Hufflepuff", color="yellow")
    plt.scatter(by_house["Ravenclaw"]["x"], by_house["Ravenclaw"]["y"], alpha=0.6, label="Ravenclaw", color="blue")
    plt.scatter(by_house["Slytherin"]["x"], by_house["Slytherin"]["y"], alpha=0.6, label="Slytherin", color="green")

    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title(f"{feature_1} vs {feature_2}")
    plt.legend()
    plt.tight_layout()
    plt.show()