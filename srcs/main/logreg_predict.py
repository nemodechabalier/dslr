import argparse
import sys
from pathlib import Path
import json

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data import try_prepare_dataset
from ml import load_json, predict_houses


def main() -> int:
    """Entry point for generating Hogwarts house predictions."""

    parser = argparse.ArgumentParser(description="One-vs-all logistic regression prediction entrypoint.")
    parser.add_argument("datasets", nargs="?", default="datasets/dataset_test.csv")
    parser.add_argument("logreg_weights", nargs="?", default="datasets/logreg_weights.json", help="Path to features, trained weights and describe file")
    parser.add_argument("--output", nargs="?", default="houses.csv", help="Prediction CSV output path")
    args = parser.parse_args()

    dataset_store = try_prepare_dataset([args.datasets, f"../{args.datasets}", f"./{args.datasets}"])
    if dataset_store is None:
        print("Unable to load dataset.")
        return 1

    data = load_json(args.logreg_weights)
    predict_houses(dataset_store, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
