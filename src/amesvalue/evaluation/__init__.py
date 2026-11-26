"""Train, select and evaluate the benchmark suite in historical dollars."""

import argparse
import json
import logging
import os
import tomllib
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from amesvalue.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, read_data, split_data
from amesvalue.features import preprocessor
from amesvalue.models import LinearBenchmark
from amesvalue.uncertainty import conformal_radius, interval

LOG = logging.getLogger(__name__)


def suite(seed):
    """Return model prototypes and deliberately small parameter grids."""
    candidates = {
        "Mean baseline": (make_pipeline(preprocessor(), DummyRegressor()), {}),
        "Median baseline": (
            make_pipeline(preprocessor(), DummyRegressor(strategy="median")),
            {},
        ),
        "OLS": (LinearBenchmark(transform="raw"), {}),
        "Log OLS naive": (LinearBenchmark(transform="naive"), {}),
        "Log OLS smearing": (LinearBenchmark(), {}),
        "Log OLS normal correction": (LinearBenchmark(transform="normal"), {}),
        "Stepwise AIC": (LinearBenchmark(method="aic"), {}),
        "Backward BIC": (LinearBenchmark(method="bic"), {}),
        "Interaction": (LinearBenchmark(method="interaction"), {}),
    }
    for name, reg, alphas in [
        ("Ridge", Ridge(), [0.5, 5, 30]),
        ("Lasso", Lasso(max_iter=10000), [0.00025, 0.001, 0.004]),
        ("Elastic Net", ElasticNet(max_iter=10000), [0.00025, 0.001, 0.004]),
    ]:
        candidates[name] = (
            LinearBenchmark(estimator=reg),
            {"estimator__alpha": alphas},
        )
    trees = [
        (
            "Random Forest",
            RandomForestRegressor(n_estimators=120, random_state=seed),
            {"min_samples_leaf": [1, 3]},
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(n_estimators=120, random_state=seed),
            {"min_samples_leaf": [1, 3]},
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(random_state=seed),
            {"max_leaf_nodes": [15, 31]},
        ),
    ]
    try:
        from xgboost import XGBRegressor
    except (ImportError, OSError, RuntimeError) as exc:
        LOG.warning("XGBoost unavailable: %s", exc)
    else:
        trees.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=350, learning_rate=0.05, n_jobs=1, random_state=seed
                ),
                {"max_depth": [3, 5]},
            )
        )
    for name, reg, params in trees:
        candidates[name] = (
            make_pipeline(preprocessor(), reg),
            {f"{type(reg).__name__.lower()}__{p}": v for p, v in params.items()},
        )
    return candidates


def cross_validation(train, temporal, folds):
    if not temporal:
        return list(GroupKFold(folds).split(train, groups=train.parcel))
    period = (train.sale_year * 100 + train.sale_month).to_numpy()
    months = np.unique(period)
    splits = []
    for before, after in TimeSeriesSplit(folds).split(months):
        validation = np.isin(period, months[after])
        fit = np.isin(period, months[before]) & ~train.parcel.isin(
            train.parcel[validation]
        )
        splits.append((np.flatnonzero(fit), np.flatnonzero(validation)))
    return splits


def metrics(y, pred):
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "r2": r2_score(y, pred),
        "median_ae": median_absolute_error(y, pred),
        "median_ape": 100 * np.median(np.abs(np.asarray(y) - pred) / np.asarray(y)),
    }


def infer(linear, train):
    """Fit statsmodels OLS on the benchmark's training design."""
    x = sm.add_constant(linear.design(train[FEATURES]), has_constant="add")
    return sm.OLS(np.log(train[TARGET].to_numpy()), x).fit()


def audit_groups(test, pred, low, high, train):
    frame = test.assign(
        error=np.abs(test[TARGET] - pred), covered=test[TARGET].between(low, high)
    )
    for col, name in [(TARGET, "price_range"), ("living_area", "property_size")]:
        bounds = np.unique(train[col].quantile([0.25, 0.5, 0.75]))
        frame[name] = pd.cut(frame[col], [-np.inf, *bounds, np.inf]).astype(str)
    tables = []
    for group in ["neighborhood", "overall_quality", "price_range", "property_size"]:
        table = frame.groupby(group).agg(
            n=("error", "size"),
            mae=("error", "mean"),
            median_ae=("error", "median"),
            coverage=("covered", "mean"),
        )
        table.index.name = "level"
        tables.append(
            table.reset_index().assign(group=group, sparse=lambda x: x.n < 20)
        )
    return pd.concat(tables, ignore_index=True)


def run(frame, output, seed, folds, coverage, temporal):
    name = "chronological" if temporal else "random"
    sets = split_data(frame, seed, temporal)
    train, cal, test = (sets[k] for k in ("train", "calibration", "test"))
    for key, partition in sets.items():
        partition[["original_row"]].to_csv(
            output / f"{name}_{key}_rows.csv", index=False
        )
    cv = cross_validation(train, temporal, folds)
    models, rows = {}, []
    for label, (model, parameters) in suite(seed).items():
        start = perf_counter()
        search = GridSearchCV(
            model,
            parameters,
            cv=cv,
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )
        search.fit(train[FEATURES], train[TARGET])
        models[label] = search.best_estimator_
        rows.append(
            {
                "model": label,
                "cv_mae": -search.best_score_,
                "cv_mae_sd": search.cv_results_["std_test_score"][search.best_index_],
                "parameters": json.dumps(search.best_params_),
                "training_seconds": perf_counter() - start,
            }
        )
        LOG.info("%s | %s | CV MAE %.0f", name, label, -search.best_score_)
    rows.sort(key=lambda x: x["cv_mae"])
    winner = rows[0]["model"]
    selected = models[winner]
    radius = conformal_radius(cal[TARGET], selected.predict(cal[FEATURES]), coverage)

    # Use calibration, not test, for model explanations and sensitivity reporting.
    importance = permutation_importance(
        selected,
        cal[FEATURES],
        cal[TARGET],
        n_repeats=8,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "mae_increase": importance.importances_mean,
            "repeat_sd": importance.importances_std,
        }
    ).sort_values("mae_increase", ascending=False).to_csv(
        output / f"{name}_importance.csv", index=False
    )

    benchmark = models["Stepwise AIC"]
    fit = infer(benchmark, train)
    robust = fit.get_robustcov_results(cov_type="HC3")
    terms = ["Intercept", *benchmark.column_names_]
    ci = robust.conf_int()
    std = np.r_[np.nan, benchmark.design(train[FEATURES]).std(axis=0)]
    pd.DataFrame(
        {
            "term": terms,
            "coefficient": fit.params,
            "p_value_hc3": robust.pvalues,
            "lower": ci[:, 0],
            "upper": ci[:, 1],
            "percent": 100 * np.expm1(fit.params),
            "standardized": fit.params * std,
        }
    ).to_csv(output / f"{name}_coefficients.csv", index=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        influence = fit.get_influence().cooks_distance[0]
    ranked = np.argsort(np.where(np.isfinite(influence), influence, -np.inf))
    keep = ranked[:-3]
    trim = clone(benchmark).fit(train.iloc[keep][FEATURES], train.iloc[keep][TARGET])
    sensitivity = {
        "full_calibration_mae": mean_absolute_error(
            cal[TARGET], benchmark.predict(cal[FEATURES])
        ),
        "trimmed_calibration_mae": mean_absolute_error(
            cal[TARGET], trim.predict(cal[FEATURES])
        ),
        "removed_original_rows": train.iloc[ranked[-3:]].original_row.tolist(),
    }

    predictions = test[["original_row", TARGET]].copy()
    for row in rows:
        label = row["model"]
        start = perf_counter()
        pred = np.asarray(models[label].predict(test[FEATURES]), dtype=np.float64)
        row.update(
            metrics(test[TARGET], pred),
            selected=label == winner,
            inference_us_per_record=1e6 * (perf_counter() - start) / len(test),
            split=name,
        )
        predictions[label] = pred

    pred = predictions[winner].to_numpy()
    low, high = interval(pred, radius)
    predictions.assign(lower=low, upper=high).to_csv(
        output / f"{name}_predictions.csv", index=False
    )
    audit_groups(test, pred, low, high, train).to_csv(
        output / f"{name}_subgroups.csv", index=False
    )
    classical = fit.get_prediction(
        sm.add_constant(benchmark.design(test[FEATURES]), has_constant="add")
    ).summary_frame(alpha=1 - coverage)
    # Residual bootstrap around the fitted winner, using independent calibration
    # errors. This estimates a marginal additive error distribution.
    rng = np.random.default_rng(seed)
    residuals = cal[TARGET].to_numpy() - selected.predict(cal[FEATURES])
    bootstrap = rng.choice(residuals, size=20000, replace=True)
    tail = (1 - coverage) / 2
    q_low, q_high = np.quantile(bootstrap, [tail, 1 - tail])
    intervals = []
    for method, lower, upper in [
        ("Split conformal", low, high),
        (
            "Classical log OLS",
            np.exp(classical.obs_ci_lower.to_numpy()),
            np.exp(classical.obs_ci_upper.to_numpy()),
        ),
        (
            "Calibration residual bootstrap",
            np.maximum(pred + q_low, 0),
            np.maximum(pred + q_high, 0),
        ),
    ]:
        intervals.append(
            {
                "method": method,
                "nominal": coverage,
                "observed": float(test[TARGET].between(lower, upper).mean()),
                "mean_width": float(np.mean(upper - lower)),
            }
        )
    metadata = {
        "model": winner,
        "rows": {key: len(value) for key, value in sets.items()},
        "intervals": intervals,
        "sensitivity": sensitivity,
        "seed": seed,
    }
    (output / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2))
    if temporal:
        # Persist only the serving winner. Historical comparison rows are a local data product.
        categories = {c: sorted(frame[c].dropna().unique()) for c in CATEGORICAL}
        defaults = {c: int(train[c].median()) for c in NUMERIC}
        defaults.update({c: str(train[c].mode().iloc[0]) for c in CATEGORICAL})
        defaults["first_floor_area"] = min(
            defaults["first_floor_area"], defaults["living_area"]
        )
        joblib.dump(
            {
                "model": selected,
                "name": winner,
                "radius": radius,
                "coverage": coverage,
                "levels": categories,
                "defaults": defaults,
                "split": name,
            },
            output / "model.joblib.tmp",
            compress=3,
        )
        os.replace(output / "model.joblib.tmp", output / "model.joblib")
        frame[FEATURES + [TARGET, "sale_year"]].to_csv(
            output / "historical_sales.csv", index=False
        )
        summary = frame.groupby("neighborhood", as_index=False).agg(
            sales=(TARGET, "size"),
            median_price=(TARGET, "median"),
            mean_living_area=("living_area", "mean"),
            mean_quality=("overall_quality", "mean"),
        )
        summary.to_csv(output / "neighborhood_summary.csv", index=False)
        counts, edges = np.histogram(frame[TARGET], bins=32)
        pd.DataFrame(
            {"price_midpoint": (edges[:-1] + edges[1:]) / 2, "sales": counts}
        ).to_csv(output / "sale_price_distribution.csv", index=False)
        (output / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "sales": len(frame),
                    "median_sale_price": float(frame[TARGET].median()),
                },
                indent=2,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/AmesHousing.xls"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.toml"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = tomllib.loads(args.config.read_text())
    frame, audit = read_data(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    with threadpool_limits(limits=1):
        rows = run(frame, args.output, temporal=False, **config)
        rows += run(frame, args.output, temporal=True, **config)
    pd.DataFrame(rows).to_csv(args.output / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
