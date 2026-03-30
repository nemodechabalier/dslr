from data.models import DatasetStore
import matplotlib.pyplot as plt
import math

def is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)

def run_histogram(dataset_store: DatasetStore, feature: str) -> None:
    """Plot score distributions for one feature split by Hogwarts house."""
    if feature not in dataset_store.feature_names:
        available = ", ".join(dataset_store.feature_names)
        raise ValueError(f"Unknown feature '{feature}'. Available features: {available}")

    plt.figure(figsize=(10, 6))
    plt.title(f"Histogramme de {feature} par maison")
    plt.xlabel(feature)
    plt.ylabel("Nombre d'eleves")

    feature_idx = dataset_store.feature_names.index(feature)

    by_house = {
        "Gryffindor": [],
        "Hufflepuff": [],
        "Ravenclaw": [],
        "Slytherin": [],
    }

    for house, row in zip(dataset_store.houses, dataset_store.raw_features):
        if house not in by_house:
            continue

        value = row[feature_idx]
        if isinstance(value, (int, float)) and not is_nan(value):
            by_house[house].append(value)

    bins = 20
    plt.hist(by_house["Gryffindor"], bins=bins, alpha=0.5, label="Gryffindor")
    plt.hist(by_house["Hufflepuff"], bins=bins, alpha=0.5, label="Hufflepuff")
    plt.hist(by_house["Ravenclaw"], bins=bins, alpha=0.5, label="Ravenclaw")
    plt.hist(by_house["Slytherin"], bins=bins, alpha=0.5, label="Slytherin")

    plt.legend()
    plt.tight_layout()
    plt.show()
