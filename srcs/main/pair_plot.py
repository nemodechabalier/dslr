import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data import try_prepare_dataset
from visualization import run_pair_plot



def main() -> int:
    """Entry point for the pair plot visualization command."""

    parser = argparse.ArgumentParser(
        description="Display a pair plot for selected features split by Hogwarts house.",
    )
    parser.add_argument(
        "datasets",
        nargs="?",
        default="datasets/dataset_train.csv",
        help="Path to CSV dataset file (example: datasets/dataset_train.csv)",
    )
    parser.add_argument(
        "features",
        nargs="*",
        default=["Ancient Runes", "Arithmancy", "Astronomy","Care of Magical Creatures", "Charms", "Defense Against the Dark Arts", 
                 "Divination","Flying", "Herbology","History of Magic", "Muggle Studies", "Potions", "Transfiguration"],
        help=(
            "Feature names to include in pair plot "
            "(example: Arithmancy Astronomy Herbology)"
        ),
    )
    args = parser.parse_args()

    paths = [args.datasets, f"../{args.datasets}", f"./{args.datasets}"]
    dataset_store = try_prepare_dataset(paths)
    if dataset_store is None:
        print(
            "Error: unable to load dataset. "
            "Check the path and file format. Tried paths: "
            + ", ".join(paths)
        )
        return 1

    try:
        run_pair_plot(dataset_store, args.features)
    except ValueError as err:
        print(f"Error: {err}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

