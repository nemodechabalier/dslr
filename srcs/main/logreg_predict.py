import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data.pipeline import try_prepare_dataset
from ml.predict import load_weights, predict_houses, write_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="One-vs-all logistic regression prediction entrypoint.")
    parser.add_argument("dataset", nargs="?", default="dataset/dataset_test.csv")
    parser.add_argument("weights", help="Path to trained weights file")
    parser.add_argument("--output", default="houses.csv", help="Prediction CSV output path")
    args = parser.parse_args()

    dataset_store = try_prepare_dataset([args.dataset, f"../{args.dataset}", f"./{args.dataset}"])
    if dataset_store is None:
        print("Unable to load dataset.")
        return 1

    weights = load_weights(args.weights)
    predictions = predict_houses(dataset_store, weights)
    write_predictions(predictions, args.output)
    print(f"Predictions saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
