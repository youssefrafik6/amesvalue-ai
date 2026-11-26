"""Render a portable HTML comparison chart from saved evaluation metrics."""

import argparse
from pathlib import Path

import pandas as pd

from amesvalue.visualization import bars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    options = parser.parse_args()
    comparison = pd.read_csv(options.results / "model_comparison.csv")
    chart = bars(
        comparison,
        x="model",
        y="mae",
        facet_col="split",
        color="selected",
        title="Held-out Ames sale price error",
        labels={"mae": "MAE ($)", "model": ""},
    )
    chart.write_html(options.results / "model_comparison.html")


if __name__ == "__main__":
    main()
