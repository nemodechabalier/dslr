import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data.pipeline import try_prepare_dataset
from visualization.histogram import run_histogram


def main() -> int:
    parser = argparse.ArgumentParser(description="Histogram script entrypoint.")
    parser.add_argument("dataset", nargs="?", default="dataset/dataset_train.csv")
    args = parser.parse_args()

    dataset_store = try_prepare_dataset([args.dataset, f"../{args.dataset}", f"./{args.dataset}"])
    if dataset_store is None:
        print("Unable to load dataset.")
        return 1

    run_histogram(dataset_store, "Arithmancy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
