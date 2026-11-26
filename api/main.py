"""AmesValue AI historical sale-price service."""

import logging
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException

from amesvalue.serving import DISCLAIMER, Engine, Prediction, Property, load_engine

app = FastAPI(title="AmesValue AI", version="1.0.0", description=DISCLAIMER)
log = logging.getLogger(__name__)


def engine() -> Engine:
    try:
        return load_engine()
    except (FileNotFoundError, OSError) as exc:
        log.exception("Model artifact could not be loaded")
        raise HTTPException(
            503, "Model artifact unavailable. Run training first."
        ) from exc


Service = Annotated[Engine, Depends(engine)]
Batch = Annotated[list[Property], Body(min_length=1, max_length=100)]
Comparison = Annotated[list[Property], Body(min_length=2, max_length=10)]


def predict(service, properties):
    try:
        return service.predict(properties)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/health")
def health(service: Service):
    return {"status": "ready", "model": service.bundle["name"]}


@app.get("/model-info")
def model_info(service: Service):
    return {
        key: service.bundle[key]
        for key in ("name", "coverage", "levels", "defaults", "split")
    }


@app.post("/predict", response_model=Prediction)
@app.post("/predict-interval", response_model=Prediction)
def predict_one(property_: Property, service: Service):
    return predict(service, [property_])[0]


@app.post("/batch-predict", response_model=list[Prediction])
def batch_predict(properties: Batch, service: Service):
    return predict(service, properties)


@app.post("/compare-properties")
def compare_properties(properties: Comparison, service: Service):
    predictions = predict(service, properties)
    return {
        "predictions": predictions,
        "difference_from_first": [
            p.estimate - predictions[0].estimate for p in predictions
        ],
        "drivers": [service.explain(p)[:5] for p in properties],
    }


@app.get("/model-metrics")
def model_metrics(service: Service):
    return service.table("model_comparison").to_dict("records")


@app.get("/feature-importance")
def importance(service: Service):
    return service.table(f"{service.bundle['split']}_importance").to_dict("records")


@app.get("/neighborhood-summary")
def neighborhoods(service: Service):
    return service.neighborhood_summary().to_dict("records")
