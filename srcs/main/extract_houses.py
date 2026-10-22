import argparse
import csv
import sys
from pathlib import Path
from typing import List

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data import try_prepare_dataset

def write_houses(houses: List[str], output_path: str = "houses.csv") -> None:
    """Write predicted houses to a CSV file with the required header."""

    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in enumerate(houses):
            writer.writerow([index, house])

def main() -> int:
    """Entry point for extracting houses from dataset."""
    parser = argparse.ArgumentParser(description="extract houses from dataset")
    parser.add_argument("datasets", nargs="?", default="datasets/dataset_train.csv")
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
        write_houses(dataset_store.houses)
    except ValueError as err:
        print(f"Error: {err}")
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())