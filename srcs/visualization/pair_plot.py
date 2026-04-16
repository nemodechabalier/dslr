from data.models import DatasetStore
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data.models import DatasetStore
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn

def is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def run_pair_plot(dataset_store: DatasetStore, features: list) -> None:
    """Placeholder for pair plot visualization.

    Expected question:
    Which features should be selected for logistic regression?
    """
    if len(features) < 2:
        raise ValueError("Pair plot requires at least 2 features.")

    for feature in features:
        if feature not in dataset_store.feature_names:
            available = ", ".join(dataset_store.feature_names)
            raise ValueError(
                f"Feature '{feature}' not found in dataset. "
                f"Available features are: {available}"
            )

    selected_column = features + ["Hogwarts House"]
    data = dataset_store.raw_dataframe[selected_column].copy()
    data = data.dropna(subset=features)
    
    if data.empty:
        raise ValueError("No rows available to plot after filtering missing values.")


    house_palette = {
        "Gryffindor": "#C62828",
        "Hufflepuff": "#F9A825",
        "Ravenclaw": "#1565C0",
        "Slytherin": "#2E7D32",
    }

    sns.pairplot(
        data=data,
        vars=features,
        hue="Hogwarts House",
        palette=house_palette,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 18},
    )
    plt.show()
    
