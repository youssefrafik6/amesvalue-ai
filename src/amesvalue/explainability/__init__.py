"""Model-agnostic dollar sensitivities and historical comparables."""

import pandas as pd

from amesvalue.data import FEATURES


def local_changes(model, record, reference):
    """Replace one input at a time with the training reference, holding the rest fixed."""
    variants = [record, *[{**record, key: reference[key]} for key in FEATURES]]
    values = model.predict(pd.DataFrame(variants))
    return sorted(
        [
            {"feature": key, "difference": float(values[0] - value)}
            for key, value in zip(FEATURES, values[1:], strict=True)
        ],
        key=lambda row: abs(row["difference"]),
        reverse=True,
    )


def comparables(sales, record, n=5):
    """Rank sales within neighborhood by scaled differences in area and quality."""
    eligible = sales[sales.neighborhood == record["neighborhood"]].copy()
    distance = (
        ((eligible.living_area - record["living_area"]) / 500) ** 2
        + (eligible.overall_quality - record["overall_quality"]) ** 2
        + ((eligible.year_built - record["year_built"]) / 30) ** 2
    )
    return eligible.loc[
        distance.nsmallest(n).index,
        ["sale_year", "overall_quality", "living_area", "year_built", "sale_price"],
    ]
