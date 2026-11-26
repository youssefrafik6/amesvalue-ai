"""Independently recompute holdout scores for an existing training run."""

import argparse
from pathlib import Path

import pandas as pd

from amesvalue.evaluation import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    options = parser.parse_args()
    scores = pd.read_csv(options.results / "model_comparison.csv")
    for split, group in scores.groupby("split"):
        predictions = pd.read_csv(options.results / f"{split}_predictions.csv")
        for row in group.itertuples():
            computed = metrics(predictions.sale_price, predictions[row.model])
            for key, value in computed.items():
                if abs(value - getattr(row, key)) > 1e-7:
                    raise ValueError(
                        f"{split} {row.model} {key} differs from saved predictions"
                    )
        print(f"{split}: verified {len(group)} models on {len(predictions)} test sales")


if __name__ == "__main__":
    main()
