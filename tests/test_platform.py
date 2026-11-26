"""Synthetic fixtures exercise the contract without distributing course records."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from amesvalue.data import FEATURES, read_data, split_data
from amesvalue.models import LinearBenchmark
from amesvalue.serving import Engine, Property, load_engine
from amesvalue.uncertainty import conformal_radius
from api.main import app

HOME = {
    "overall_quality": 6,
    "living_area": 1500,
    "basement_area": 900,
    "garage_area": 400,
    "year_built": 1980,
    "year_remodeled": 1990,
    "first_floor_area": 1000,
    "lot_area": 9000,
    "full_baths": 2,
    "total_rooms": 6,
    "neighborhood": "NAmes",
    "house_style": "1Story",
    "building_type": "1Fam",
}


@pytest.fixture
def trained(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                **HOME,
                "overall_quality": i % 9 + 1,
                "living_area": 1200 + i * 20,
                "neighborhood": "Blmngtn" if i % 3 else "NAmes",
            }
            for i in range(32)
        ]
    )
    target = 100000 + 12000 * frame.overall_quality + 40 * frame.living_area
    model = LinearBenchmark(method="aic").fit(frame, target)
    joblib.dump(
        {
            "model": model,
            "name": "Stepwise AIC",
            "coverage": 0.9,
            "radius": 15000,
            "defaults": HOME,
            "split": "random",
            "levels": {
                "neighborhood": ["Blmngtn", "NAmes"],
                "house_style": ["1Story"],
                "building_type": ["1Fam"],
            },
        },
        tmp_path / "model.joblib",
    )
    for name in ["model_comparison", "neighborhood_summary", "random_importance"]:
        pd.DataFrame([{"x": 1}]).to_csv(tmp_path / f"{name}.csv", index=False)
    monkeypatch.setenv("AMES_RESULTS", str(tmp_path))
    load_engine.cache_clear()
    yield Engine(tmp_path)
    load_engine.cache_clear()


def test_request_contract(trained):
    with TestClient(app) as client:
        for route in [
            "health",
            "model-info",
            "model-metrics",
            "feature-importance",
            "neighborhood-summary",
        ]:
            assert client.get(f"/{route}").is_success
        expected = trained.predict([Property(**HOME)])[0].model_dump()
        assert client.post("/predict", json=HOME).json() == expected
        assert client.post("/predict-interval", json=HOME).json() == expected
        assert client.post("/batch-predict", json=[HOME, HOME]).json() == [
            expected,
            expected,
        ]
        better = {**HOME, "overall_quality": 8}
        pair = client.post("/compare-properties", json=[HOME, better])
        assert pair.is_success
        assert pair.json()["difference_from_first"][1] > 0
        for invalid in [
            {**HOME, "neighborhood": "unknown"},
            {**HOME, "overall_quality": 11},
            {**HOME, "year_remodeled": 1970},
            {**HOME, "first_floor_area": 1900},
            {k: v for k, v in HOME.items() if k != "garage_area"},
        ]:
            assert client.post("/predict", json=invalid).status_code == 422
        assert client.post("/batch-predict", json=[HOME] * 101).status_code == 422
        assert client.post("/compare-properties", json=[HOME]).status_code == 422


def test_conformal_order_statistic():
    assert conformal_radius(np.arange(1, 10), np.zeros(9), 0.9) == 9
    with pytest.raises(ValueError):
        conformal_radius([1], [0], 0.9)


@pytest.mark.parametrize(
    "y,prediction,coverage",
    [([1, 2], [[1], [2]], 0.5), ([1, 2], [1, np.nan], 0.5), ([1, 2], [1, 2], np.nan)],
)
def test_conformal_rejects_invalid_calibration(y, prediction, coverage):
    with pytest.raises(ValueError):
        conformal_radius(y, prediction, coverage)


def test_disjoint_parcels_and_months():
    frame = pd.DataFrame(
        {
            "parcel": np.repeat(range(36), 2),
            "sale_year": [2008] * 24 + [2009] * 24 + [2010] * 24,
            "sale_month": np.tile(np.repeat(range(1, 13), 2), 3),
        }
    )
    for temporal in (False, True):
        split = split_data(frame, 326, temporal)
        parts = list(split.values())
        for i, part in enumerate(parts):
            for next_part in parts[i + 1 :]:
                assert not set(part.parcel) & set(next_part.parcel)
                if temporal:
                    assert (part.sale_year * 100 + part.sale_month).max() < (
                        next_part.sale_year * 100 + next_part.sale_month
                    ).min()


def test_workbook_contract_when_available():
    workbook = Path("data/raw/AmesHousing.xls")
    if not workbook.exists():
        pytest.skip("Original workbook is provided locally")
    frame, audit = read_data(workbook)
    assert audit["rows"] == 2930 and audit["columns"] == 82
    assert set(FEATURES) <= set(frame)
    assert frame.loc[1498, "living_area"] == 5642


def test_dashboard_pages_when_artifacts_available():
    if not Path("results/historical_sales.csv").exists():
        pytest.skip("Record-level training artifacts are optional in CI")
    load_engine.cache_clear()
    app_test = AppTest.from_file(
        Path("dashboard/app.py").resolve(), default_timeout=20
    ).run()
    assert not app_test.exception
    for page in app_test.sidebar.radio[0].options:
        app_test.sidebar.radio[0].set_value(page).run()
        assert not app_test.exception, page
        if page in ["Home price predictor", "Property comparison", "What-if analysis"]:
            app_test.button[0].click().run()
            assert not app_test.exception and not app_test.error, page
    load_engine.cache_clear()


def test_public_artifacts_predict_without_workbook(tmp_path, monkeypatch):
    from shutil import copy2

    clean = tmp_path / "public_results"
    clean.mkdir()
    for name in [
        "model.joblib",
        "model_comparison.csv",
        "dataset_summary.json",
        "chronological_metadata.json",
        "chronological_importance.csv",
        "sale_price_distribution.csv",
    ]:
        copy2(Path("results") / name, clean / name)
    monkeypatch.setenv("AMES_RESULTS", str(clean))
    load_engine.cache_clear()
    app_test = AppTest.from_file(
        Path("dashboard/app.py").resolve(), default_timeout=20
    ).run()
    assert not app_test.exception
    app_test.sidebar.radio[0].set_value("Home price predictor").run()
    app_test.button[0].click().run()
    assert not app_test.exception and not app_test.error
    assert app_test.metric[0].value.startswith("$")
    assert app_test.metric[3].value == "Extra Trees"
    load_engine.cache_clear()
