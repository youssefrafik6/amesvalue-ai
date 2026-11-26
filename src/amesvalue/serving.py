"""Shared prediction service and HTTP/UI input contract."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from amesvalue.data import CATEGORICAL
from amesvalue.explainability import local_changes
from amesvalue.uncertainty import interval

DISCLAIMER = (
    "This project is an educational machine learning demonstration using historical Ames Housing "
    "data. It is not a professional appraisal, mortgage valuation, or investment recommendation."
)


class Property(BaseModel):
    """Attributes of a hypothetical property in historical Ames."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    overall_quality: int = Field(ge=1, le=10)
    living_area: float = Field(gt=0, le=6000)
    basement_area: float = Field(ge=0, le=6500)
    garage_area: float = Field(ge=0, le=1600)
    year_built: int = Field(ge=1870, le=2010)
    year_remodeled: int = Field(ge=1870, le=2010)
    first_floor_area: float = Field(gt=0, le=6000)
    lot_area: float = Field(gt=0, le=220000)
    full_baths: int = Field(ge=0, le=5)
    total_rooms: int = Field(ge=1, le=20)
    neighborhood: str = Field(min_length=1, max_length=20)
    house_style: str = Field(min_length=1, max_length=20)
    building_type: str = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def consistency(self):
        if self.year_remodeled < self.year_built:
            raise ValueError("Remodel year precedes construction year")
        if self.first_floor_area > self.living_area:
            raise ValueError("First-floor area exceeds living area")
        if self.first_floor_area > self.lot_area:
            raise ValueError("First-floor area exceeds lot area")
        return self


class Prediction(BaseModel):
    estimate: float
    lower: float
    upper: float
    nominal_coverage: float
    model: str


class Engine:
    """Load a trusted local training artifact and share its inference contract."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.bundle = joblib.load(directory / "model.joblib")

    def records(self, properties: list[Property]) -> list[dict]:
        records = [item.model_dump() for item in properties]
        for record in records:
            for key in CATEGORICAL:
                if record[key] not in self.bundle["levels"][key]:
                    raise ValueError(f"Unsupported {key}: {record[key]}")
        return records

    def predict(self, properties: list[Property]) -> list[Prediction]:
        records = self.records(properties)
        prices = self.bundle["model"].predict(pd.DataFrame(records)).clip(min=0)
        lower, upper = interval(prices, self.bundle["radius"])
        return [
            Prediction(
                estimate=price,
                lower=lo,
                upper=hi,
                nominal_coverage=self.bundle["coverage"],
                model=self.bundle["name"],
            )
            for price, lo, hi in zip(prices, lower, upper, strict=True)
        ]

    def explain(self, property_: Property) -> list[dict]:
        return local_changes(
            self.bundle["model"], self.records([property_])[0], self.bundle["defaults"]
        )

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.directory / f"{name}.csv")

    def neighborhood_summary(self) -> pd.DataFrame:
        return self.table("neighborhood_summary")


@lru_cache(maxsize=1)
def load_engine() -> Engine:
    return Engine(Path(os.getenv("AMES_RESULTS", "results")))
