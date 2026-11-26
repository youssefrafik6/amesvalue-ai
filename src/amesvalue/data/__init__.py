"""Workbook contract and deterministic, parcel-aware data partitions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

COLUMNS = {
    "Overall Qual": "overall_quality",
    "Gr Liv Area": "living_area",
    "Total Bsmt SF": "basement_area",
    "Garage Area": "garage_area",
    "Year Built": "year_built",
    "Year Remod/Add": "year_remodeled",
    "1st Flr SF": "first_floor_area",
    "Lot Area": "lot_area",
    "Full Bath": "full_baths",
    "TotRms AbvGrd": "total_rooms",
    "Neighborhood": "neighborhood",
    "House Style": "house_style",
    "Bldg Type": "building_type",
    "SalePrice": "sale_price",
    "PID": "parcel",
    "Yr Sold": "sale_year",
    "Mo Sold": "sale_month",
}
NUMERIC = list(COLUMNS.values())[:10]
CATEGORICAL = ["neighborhood", "house_style", "building_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "sale_price"


def read_data(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the original Excel file or an explicit lossless CSV conversion."""
    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    missing = COLUMNS.keys() - set(raw)
    if missing:
        raise ValueError(f"Missing workbook columns: {sorted(missing)}")
    frame = raw[list(COLUMNS)].rename(columns=COLUMNS)
    frame.insert(0, "original_row", np.arange(1, len(frame) + 1))
    numeric_columns = NUMERIC + [TARGET, "sale_year", "sale_month"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    for column in CATEGORICAL:
        frame[column] = frame[column].map(
            lambda value: (
                value.strip() if isinstance(value, str) and value.strip() else np.nan
            )
        )
    invalid = (
        ~frame[TARGET].gt(0)
        | ~np.isfinite(frame[TARGET])
        | ~frame.sale_year.between(2006, 2010)
        | ~frame.sale_month.isin(range(1, 13))
        | frame.parcel.isna()
    )
    if invalid.any():
        raise ValueError(
            f"Invalid price, sale date or parcel at rows {frame.original_row[invalid].tolist()}"
        )
    suspicious = {
        "construction_year": int(
            ((frame.year_built < 1800) | (frame.year_built > frame.sale_year)).sum()
        ),
        "remodel_year": int(
            (
                (frame.year_remodeled < frame.year_built)
                | (frame.year_remodeled > frame.sale_year)
            ).sum()
        ),
        "area": int(
            (
                (
                    frame[["basement_area", "garage_area", "living_area", "lot_area"]]
                    < 0
                ).any(axis=1)
                | (frame.first_floor_area > frame.living_area)
            ).sum()
        ),
        "quality": int((~frame.overall_quality.isin(range(1, 11))).sum()),
    }
    frame = frame.replace([np.inf, -np.inf], np.nan)
    audit = {
        "rows": len(raw),
        "columns": len(raw.columns),
        "missing_by_field": frame[FEATURES].isna().sum().to_dict(),
        "distinct_levels": {c: sorted(frame[c].dropna().unique()) for c in CATEGORICAL},
        "repeated_parcel_rows": int(frame.parcel.duplicated().sum()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "large_living_area_over_4000": int(frame.living_area.gt(4000).sum()),
        "flagged_values": suspicious,
    }
    frame = frame.loc[~raw.duplicated()].reset_index(drop=True)
    return frame, audit


def split_data(
    frame: pd.DataFrame, seed: int, temporal: bool
) -> dict[str, pd.DataFrame]:
    """Keep parcels disjoint and whole calendar months together in temporal evaluation."""
    if temporal:
        period = frame.sale_year * 100 + frame.sale_month
        test = frame[period >= 200909]
        calibration = frame[(period >= 200905) & (period < 200909)]
        calibration = calibration[~calibration.parcel.isin(test.parcel)]
        train = frame[
            (period < 200905)
            & ~frame.parcel.isin(pd.concat([test, calibration]).parcel)
        ]
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        development_idx, test_idx = next(splitter.split(frame, groups=frame.parcel))
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
        train_idx, cal_idx = next(
            splitter.split(development, groups=development.parcel)
        )
        train, calibration = development.iloc[train_idx], development.iloc[cal_idx]
    return {"train": train, "calibration": calibration, "test": test}
