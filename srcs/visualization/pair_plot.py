from data.models import DatasetStore
import matplotlib.pyplot as plt
import seaborn as sns


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

    num_features = len(features)
    size_per_plot = 2.8
    fig_size = (max(6.0, num_features * size_per_plot), max(6.0, num_features * size_per_plot))

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
        height=2.6,
        aspect=1.0,
    )

    g.fig.set_size_inches(*fig_size)

    for ax in g.axes.flat:
        if ax is not None:
            ax.tick_params(axis="both", labelsize=8)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=9, labelpad=8)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=9, labelpad=8)

    if g._legend is not None:
        g._legend.set_title("Hogwarts House")
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend._loc = 6
        g._legend.set_frame_on(False)

    g.fig.subplots_adjust(left=0.14, bottom=0.14, right=0.78, top=0.95, wspace=0.15, hspace=0.15)

    plt.savefig(f"visu/pair_plots/{'_'.join(features)}.png", bbox_inches="tight")
    
