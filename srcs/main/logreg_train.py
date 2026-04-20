import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data.pipeline import try_prepare_dataset
from ml.train import save_weights, train_one_vs_all, train_models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one-vs-all logistic regression models on selected features.",
    )
    parser.add_argument(
        "dataset",
        help="Path to CSV dataset file (example: dataset/dataset_train.csv)",
    )
    parser.add_argument(
        "features",
        nargs="*",
        default=["Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Charms", "Flying"],
        help=(
            "Feature names to use for training "
            "(defaults to Arithmancy, Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes, History of Magic, Charms, Flying if not provided)"
        ),
    )
    args = parser.parse_args()

    paths = [args.dataset, f"../{args.dataset}", f"./{args.dataset}"]
    dataset_store = try_prepare_dataset(paths)
    if dataset_store is None:
        print(
            "Error: unable to load dataset. "
            "Check the path and file format. Tried paths: "
            + ", ".join(paths)
        )
        return 1

    try:
        train_models(dataset_store, args.features)
    except ValueError as err:
        print(f"Error: {err}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
