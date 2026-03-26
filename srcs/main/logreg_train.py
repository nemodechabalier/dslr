import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data.pipeline import try_prepare_dataset
from ml.train import save_weights, train_one_vs_all


def main() -> int:
    parser = argparse.ArgumentParser(description="One-vs-all logistic regression training entrypoint.")
    parser.add_argument("dataset", nargs="?", default="dataset/dataset_train.csv")
    parser.add_argument("--weights-out", default="weights.json", help="Output path for trained weights")
    args = parser.parse_args()

    dataset_store = try_prepare_dataset([args.dataset, f"../{args.dataset}", f"./{args.dataset}"])
    if dataset_store is None:
        print("Unable to load dataset.")
        return 1

    weights = train_one_vs_all(dataset_store)
    save_weights(weights, args.weights_out)
    print(f"Weights saved to {args.weights_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
