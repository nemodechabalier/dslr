from data.models import DatasetStore
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def run_pair_plot(dataset_store: DatasetStore, features: list) -> None:
    """Render a pair plot for the selected features grouped by house."""
    if not isinstance(features, list) or len(features) < 2:
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

    g = sns.pairplot(
        data=data,
        vars=features,
        hue="Hogwarts House",
        palette=house_palette,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 18},
        height=2.4,
        aspect=1.0,
    )

    for ax in g.axes.flat:
        if ax is not None:
            ax.tick_params(axis="both", labelsize=8)

    if g._legend is not None:
        g._legend.set_title("Hogwarts House")

    out_dir = Path("visu/pair_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "_".join(feature.replace(" ", "_") for feature in features)
    # plt.show()

    plt.savefig(out_dir / f"{safe_name}.png", bbox_inches="tight")
    
