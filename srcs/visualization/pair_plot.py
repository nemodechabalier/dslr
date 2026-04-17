from data.models import DatasetStore
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_pair_plot(dataset_store: DatasetStore, features: list) -> None:
    """Display a pair plot for multiple selected features with house coloring."""
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

    # Adjust figure size based on number of features
    num_features = len(features)
    size_per_plot = 2.5
    fig_size = (num_features * size_per_plot, num_features * size_per_plot)

    house_palette = {
        "Gryffindor": "#C62828",
        "Hufflepuff": "#F9A825",
        "Ravenclaw": "#1565C0",
        "Slytherin": "#2E7D32",
    }

    g = sns.pairplot(
        data=data,
        vars=features,
        hue="Hogwarts House",
        palette=house_palette,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 18},
        height=size_per_plot / 2.5,
    )

    for ax in g.axes.flat:
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=8, rotation=45, ha="right")
            ax.set_ylabel(ax.get_ylabel(), fontsize=8, rotation=45, ha="right")

    g.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    plt.show()
    
