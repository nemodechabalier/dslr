import argparse
import sys
from pathlib import Path
from typing import Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
SRCS_DIR = CURRENT_DIR.parent
if str(SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(SRCS_DIR))

from data.pipeline import try_prepare_dataset


def _format_value(value: float) -> str:
    return f"{value:.6f}"


def _shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _print_stats(stats: Dict[str, Dict[str, float]], feature_names: List[str], title: str, show_missing: bool) -> None:
    if not stats:
        print(f"{title}: no numeric features found.")
        return

    labels = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "skewness",
        "kurtosis",
    ]
    if show_missing:
        labels.append("missing")

    feature_width = 30
    metric_width = 12

    print(f"\n{title}")
    header = f"{'Feature':<{feature_width}} " + " ".join(f"{label:>{metric_width}}" for label in labels)
    print(header)
    print("-" * len(header))

    for name in feature_names:
        if name not in stats:
            continue

        line = f"{_shorten(name, feature_width):<{feature_width}}"
        for label in labels:
            value = stats[name].get(label)
            if value is None:
                line += f" {'-':>{metric_width}}"
            else:
                line += f" {_format_value(value):>{metric_width}}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual describe for Hogwarts dataset.")
    parser.add_argument("dataset", nargs="?", default="dataset/dataset_train.csv", help="Path to dataset CSV")
    args = parser.parse_args()

    paths = [args.dataset, f"../{args.dataset}", f"./{args.dataset}"]
    dataset_store = try_prepare_dataset(paths)

    if dataset_store is None:
        print("Unable to load dataset.")
        return 1

    _print_stats(
        dataset_store.stats_raw,
        dataset_store.feature_names,
        "Stats before cleaning",
        show_missing=True,
    )
    _print_stats(
        dataset_store.stats_clean,
        dataset_store.feature_names,
        "Stats after cleaning",
        show_missing=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
