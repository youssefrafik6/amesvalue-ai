"""Streamlit dashboard for the AmesValue AI historical valuation study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError
from scipy.stats import probplot

from amesvalue.data import CATEGORICAL, FEATURES, NUMERIC
from amesvalue.explainability import comparables
from amesvalue.serving import DISCLAIMER, Property, load_engine
from amesvalue.visualization import COLOR, PALETTE, TEMPLATE, bars

st.set_page_config(page_title="AmesValue AI", page_icon="🏡", layout="wide")
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {background:#f3f6f5}
[data-testid="stSidebar"] {background:#102d2b}
[data-testid="stSidebar"] * {color:#f3faf8 !important}
[data-testid="stMetric"] {padding:1rem;border:1px solid #dce7e2;border-radius:12px;background:white}
.block-container {padding-top:2rem;max-width:1250px}
</style>
""",
    unsafe_allow_html=True,
)

PAGES = [
    "Executive overview",
    "Home price predictor",
    "Property comparison",
    "Neighborhood analytics",
    "Model explainability",
    "Statistical diagnostics",
    "Model comparison",
    "What-if analysis",
    "Research & methodology",
    "Data explorer",
]
st.sidebar.title("AmesValue AI")
page = st.sidebar.radio("Explore the project", PAGES)
st.sidebar.caption("Ames, Iowa · 2006–2010")
st.title("AmesValue AI" if page == PAGES[0] else page)
st.caption(
    "Understand What Drives Home Prices."
    if page == PAGES[0]
    else "AmesValue AI · Historical Ames sales"
)
try:
    engine = load_engine()
    comparison = engine.table("model_comparison")
    dataset = json.loads((engine.directory / "dataset_summary.json").read_text())
    try:
        sales = engine.table("historical_sales")
    except FileNotFoundError:
        sales = None
    split_name = engine.bundle["split"]
    metadata = json.loads(
        (engine.directory / f"{split_name}_metadata.json").read_text()
    )
except OSError:
    st.error("Training outputs are missing. Run the training workflow in the README.")
    st.stop()

winner = comparison[(comparison.split == split_name) & comparison.selected].iloc[0]
default = engine.bundle["defaults"]


def title(key):
    label = key.replace("_", " ").title()
    return f"{label} (sq ft)" if key.endswith("_area") else label


def dollars(value):
    return f"${value:,.0f}"


def chart(fig):
    st.plotly_chart(fig, use_container_width=True)


def property_fields(key, defaults=None):
    values = dict(default if defaults is None else defaults)
    cols = st.columns(3)
    for i, name in enumerate(FEATURES):
        with cols[i % 3]:
            if name in CATEGORICAL:
                levels = engine.bundle["levels"][name]
                values[name] = st.selectbox(
                    title(name),
                    levels,
                    index=levels.index(values[name]),
                    key=f"{key}-{name}",
                )
            else:
                fields = Property.model_fields[name].metadata
                lower = next(
                    (getattr(m, "ge", 1) for m in fields if hasattr(m, "ge")), 1
                )
                upper = next(m.le for m in fields if hasattr(m, "le"))
                values[name] = st.number_input(
                    title(name),
                    min_value=int(lower),
                    max_value=int(upper),
                    value=int(values[name]),
                    key=f"{key}-{name}",
                )
    return values


def prediction_cards(pred):
    price, lower, upper, model = st.columns(4)
    price.metric("Estimated historical sale price", dollars(pred.estimate))
    lower.metric(f"{pred.nominal_coverage:.0%} interval · lower", dollars(pred.lower))
    upper.metric(f"{pred.nominal_coverage:.0%} interval · upper", dollars(pred.upper))
    model.metric("Model", pred.model)


def driver_chart(property_, key):
    drivers = pd.DataFrame(engine.explain(property_)).head(7)
    drivers["feature"] = drivers.feature.map(title)
    fig = bars(
        drivers,
        x="difference",
        y="feature",
        orientation="h",
        labels={
            "difference": "Prediction change versus reference input ($)",
            "feature": "",
        },
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        "Each bar changes one input to the training reference. Changes can interact and do not add up to the estimate."
    )


def show_history(frame):
    st.dataframe(frame, hide_index=True, use_container_width=True)


def pair_analysis(what_if):
    count = 1 if what_if else st.selectbox("Additional properties", [1, 2, 3])
    with st.form("pair"):
        st.subheader("Starting property")
        first = property_fields("first")
        st.subheader("Modified property" if what_if else "Comparison properties")
        alternatives = []
        for i in range(count):
            with st.expander(f"Property {i + 2}", expanded=i == 0):
                alternatives.append(
                    property_fields(
                        f"alternative-{i}",
                        {
                            **default,
                            "overall_quality": min(
                                10, default["overall_quality"] + i + 1
                            ),
                        },
                    )
                )
        submit = st.form_submit_button("Compare estimates", type="primary")
    if submit:
        try:
            properties = [Property(**r) for r in [first, *alternatives]]
            preds = engine.predict(properties)
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))
            return
        base = preds[0].estimate
        table = pd.DataFrame([p.model_dump() for p in preds]).assign(
            property=[f"Property {i + 1}" for i in range(len(preds))],
            difference_from_first=[p.estimate - base for p in preds],
            percent_difference=[
                100 * (p.estimate / base - 1) if base else np.nan for p in preds
            ],
        )
        show_history(table.round(1))
        show_history(pd.DataFrame([r.model_dump() for r in properties]))
        for i, prop in enumerate(properties):
            with st.expander(f"Property {i + 1} price drivers"):
                driver_chart(prop, f"drivers-{i}")
        st.caption(
            "Differences are hypothetical model associations. Renovation returns and future sale prices may differ."
        )


if page == PAGES[0]:
    st.markdown("### Historical sale prices, explained")
    st.write(
        "Explore 2,930 Ames sales and see how a university regression study translates into a tested prediction service."
    )
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        [
            "Historical sales",
            "Median sale price",
            "Chronological test MAE",
            "Chronological test R²",
        ],
        [
            f"{dataset['sales']:,}",
            dollars(dataset["median_sale_price"]),
            dollars(winner.mae),
            f"{winner.r2:.3f}",
        ],
        strict=True,
    ):
        col.metric(label, value)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sale price distribution")
        chart(
            bars(
                engine.table("sale_price_distribution"),
                x="price_midpoint",
                y="sales",
                labels={
                    "price_midpoint": "Historical sale price ($)",
                    "sales": "Sales",
                },
            )
        )
    with right:
        st.subheader("Model price drivers")
        drivers = (
            engine.table(f"{split_name}_importance").head(7).sort_values("mae_increase")
        )
        drivers["feature"] = drivers.feature.map(title)
        chart(
            bars(
                drivers,
                x="mae_increase",
                y="feature",
                orientation="h",
                labels={"mae_increase": "Calibration MAE increase ($)", "feature": ""},
            )
        )
    st.caption(
        f"{winner.model} was selected by training-fold MAE. Chronological test size: {metadata['rows']['test']}."
    )
elif page == PAGES[1]:
    with st.form("predict"):
        record = property_fields("predict")
        submit = st.form_submit_button("Estimate sale price", type="primary")
    if submit:
        try:
            prop = Property(**record)
            pred = engine.predict([prop])[0]
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))
        else:
            prediction_cards(pred)
            observed = metadata["intervals"][0]["observed"]
            st.caption(
                f"Observed chronological test coverage was {observed:.1%}. The interval describes uncertainty for one sale under historical conditions."
            )
            st.subheader("Important price drivers")
            driver_chart(prop, "predict-drivers")
            st.subheader("Similar historical sales")
            if sales is None:
                st.info(
                    "Comparable sales are available when the workbook is supplied locally."
                )
            else:
                show_history(comparables(sales, record))
elif page == PAGES[2] or page == PAGES[7]:
    pair_analysis(page == PAGES[7])
elif page == PAGES[3]:
    levels = engine.bundle["levels"]["neighborhood"]
    chosen = st.multiselect(
        "Neighborhoods", levels, default=["NAmes", "Edwards", "NridgHt"]
    )
    st.subheader("Observed historical prices")
    show_history(
        engine.neighborhood_summary().query("neighborhood in @chosen").round(1)
    )
    if sales is not None:
        chart(
            px.box(
                sales.query("neighborhood in @chosen"),
                x="neighborhood",
                y="sale_price",
                template=TEMPLATE,
                color_discrete_sequence=PALETTE,
            )
        )
    st.subheader("Regression-adjusted associations")
    coef = engine.table("random_coefficients")
    effects = coef[coef.term.str.startswith("neighborhood_")].copy()
    effects["neighborhood"] = effects.term.str.removeprefix("neighborhood_")
    baseline_row = pd.DataFrame(
        [{"neighborhood": "Blmngtn", "percent": 0, "lower": 0, "upper": 0}]
    )
    effects[["lower", "upper"]] = 100 * np.expm1(effects[["lower", "upper"]])
    effects = pd.concat([effects, baseline_row], ignore_index=True)
    effects = effects[effects.neighborhood.isin(chosen)]
    effects = effects.assign(
        above=effects.upper - effects.percent, below=effects.percent - effects.lower
    )
    chart(
        bars(
            effects,
            x="neighborhood",
            y="percent",
            error_y="above",
            error_y_minus="below",
        )
    )
    st.caption(
        "Training Stepwise AIC associations relative to Blmngtn, holding modeled characteristics fixed. HC3 intervals are descriptive after term selection."
    )
elif page == PAGES[4]:
    st.subheader("Global permutation importance")
    chart(
        bars(
            engine.table(f"{split_name}_importance"),
            x="feature",
            y="mae_increase",
            error_y="repeat_sd",
        )
    )
    st.caption(
        "Calibration MAE increase after shuffling an input. Correlated size variables can share predictive information."
    )
    st.subheader("Stepwise AIC coefficients")
    show_history(engine.table("random_coefficients"))
    st.caption(
        "Standardized values are log-price changes per design-column standard deviation. Percentage effects use 100 × (exp(β) − 1). HC3 p-values are descriptive following selection."
    )
    st.subheader("Partial dependence and ICE")
    if sales is None:
        st.info("Supply the workbook locally to explore partial dependence and ICE.")
    else:
        feature = st.selectbox("Feature to vary", NUMERIC, format_func=title)
        sample = sales[FEATURES].dropna().sample(60, random_state=326)
        grid = np.unique(sample[feature].quantile(np.linspace(0.1, 0.9, 9)))
        amount = np.array(
            [
                engine.bundle["model"].predict(sample.assign(**{feature: value}))
                for value in grid
            ]
        )
        fig = px.line(
            x=grid,
            y=amount.mean(axis=1),
            template=TEMPLATE,
            labels={"x": title(feature), "y": "Predicted dollars"},
        )
        fig.update_traces(line_color=COLOR, line_width=4, name="Partial dependence")
        for idx in range(min(10, len(sample))):
            fig.add_scatter(
                x=grid,
                y=amount[:, idx],
                mode="lines",
                opacity=0.25,
                line={"color": "#708ea1"},
                name="ICE",
                showlegend=idx == 0,
            )
        chart(fig)
        st.caption(
            "Varying a single input can create combinations rarely observed in the dataset."
        )
    with st.form("local"):
        record = property_fields("local")
        submit = st.form_submit_button("Explain property")
    if submit:
        try:
            driver_chart(Property(**record), "explain-drivers")
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))
elif page == PAGES[5]:
    names = {
        "Initial raw OLS (2,928 rows)": "mdl_initial",
        "Cleaned log OLS (2,904 rows)": "mdl_log_full",
        "Historical Stepwise AIC (2,904 rows)": "final_model",
    }
    label = st.selectbox("Historical model", list(names))
    root = engine.directory / "original_reproduction"
    try:
        diag = pd.read_csv(root / f"{names[label]}_diagnostics.csv")
    except FileNotFoundError:
        st.info("Run the historical R workflow to generate diagnostics.")
    else:
        diag["scale_location"] = np.sqrt(np.abs(diag.standardized))
        (theory, ordered), _ = probplot(diag.residual)
        col1, col2 = st.columns(2)
        options = [
            (
                "fitted",
                "residual",
                "Residuals versus fitted",
                "Look for curvature or widening residual spread.",
            ),
            (
                "theory",
                "ordered",
                "Normal Q–Q",
                "Tail departures suggest deviations from normality.",
            ),
            (
                "fitted",
                "scale_location",
                "Scale–location",
                "Increasing spread suggests nonconstant variance.",
            ),
            (
                "leverage",
                "standardized",
                "Residuals versus leverage",
                "High leverage plus large residuals can indicate influence.",
            ),
            (
                "index",
                "cook",
                "Cook’s distance",
                "Larger values indicate stronger influence on the fitted model.",
            ),
        ]
        for i, (x, y, name, explanation) in enumerate(options):
            frame = (
                pd.DataFrame({"theory": theory, "ordered": ordered})
                if x == "theory"
                else diag.reset_index()
            )
            with col1 if i % 2 == 0 else col2:
                chart(
                    px.scatter(
                        frame,
                        x=x,
                        y=y,
                        title=name,
                        template=TEMPLATE,
                        color_discrete_sequence=PALETTE,
                    )
                )
                st.caption(explanation)
        chart(
            px.imshow(
                pd.read_csv(root / "correlation.csv", index_col=0),
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Original numeric predictor correlation",
            )
        )
        chart(
            px.line(
                pd.read_csv(root / "box_cox.csv"),
                x="lambda",
                y="log_likelihood",
                title="Box–Cox profile on cleaned raw OLS",
                template=TEMPLATE,
            )
        )
        st.caption(
            "The profile compares power transformations. λ = 0 corresponds to a log transform."
        )
        pval = pd.read_csv(root / f"{names[label]}_tests.csv").bp_pvalue.iloc[0]
        st.write(
            f"Breusch–Pagan p-value {pval:.3g}. Small values indicate evidence of nonconstant residual variance."
        )
        st.caption(
            "VIF quantifies variance inflation from collinearity in the fitted design."
        )
        show_history(pd.read_csv(root / f"{names[label]}_vif.csv"))
elif page == PAGES[6]:
    split = st.selectbox("Evaluation split", ["random", "chronological"])
    table = comparison.query("split == @split")
    metric = st.selectbox(
        "Chart metric",
        ["mae", "rmse", "r2", "cv_mae", "training_seconds", "inference_us_per_record"],
    )
    chart(bars(table, x="model", y=metric, color="selected"))
    show_history(table.round(3))
    st.caption(
        "Selection uses mean validation MAE across training folds. Each split evaluates all candidates on identical test records."
    )
    try:
        predicted = engine.table(f"{split}_predictions")
    except FileNotFoundError:
        st.info(
            "Record-level comparisons are available with locally generated training outputs."
        )
    else:
        models = st.multiselect(
            "Test predictions", table.model, default=table.model.head(2).tolist()
        )
        if models:
            plot = predicted.melt(
                id_vars=["sale_price"],
                value_vars=models,
                value_name="prediction",
                var_name="model",
            )
            chart(
                px.scatter(
                    plot,
                    x="sale_price",
                    y="prediction",
                    color="model",
                    opacity=0.4,
                    template=TEMPLATE,
                )
            )
    intervals = json.loads((engine.directory / f"{split}_metadata.json").read_text())[
        "intervals"
    ]
    st.subheader("Prediction interval performance")
    show_history(pd.DataFrame(intervals).round(3))
    st.subheader("Test error and interval coverage by group")
    show_history(engine.table(f"{split}_subgroups").round(3))
elif page == PAGES[8]:
    report = Path("data/raw/MAT326_Final_Report.pdf")
    if report.is_file():
        st.download_button(
            "Download the original MAT 326 report",
            report.read_bytes(),
            file_name=report.name,
            mime="application/pdf",
        )
    for path in ["original_research.md", "methodology.md", "limitations.md"]:
        st.markdown((Path("docs") / path).read_text())
else:
    if sales is None:
        st.info(
            "The sales explorer requires a locally supplied workbook and generated training outputs."
        )
        st.stop()
    filters = st.columns(3)
    frame = sales
    for i, feature in enumerate(CATEGORICAL):
        with filters[i]:
            chosen = st.multiselect(
                title(feature), sorted(sales[feature].dropna().unique())
            )
        if chosen:
            frame = frame[frame[feature].isin(chosen)]
    sliders = st.columns(4)
    for i, feature in enumerate(
        ["sale_price", "overall_quality", "living_area", "year_built"]
    ):
        minimum, maximum = int(sales[feature].min()), int(sales[feature].max())
        with sliders[i]:
            lo, hi = st.slider(title(feature), minimum, maximum, (minimum, maximum))
        frame = frame[frame[feature].between(lo, hi)]
    st.write(f"{len(frame):,} historical sales match the filters.")
    if not frame.empty:
        options = [
            "Sale price distribution",
            "Living area",
            "Overall quality",
            "Neighborhood",
            "Correlation",
            "Year built",
            "House style",
            "Building type",
        ]
        selection = st.selectbox("Chart", options)
        idx = options.index(selection)
        if idx == 0:
            chart(px.histogram(frame, x="sale_price", template=TEMPLATE))
        elif idx == 4:
            chart(
                px.imshow(
                    frame[NUMERIC + ["sale_price"]].corr(),
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu_r",
                )
            )
        else:
            fields = {
                1: "living_area",
                2: "overall_quality",
                3: "neighborhood",
                5: "year_built",
                6: "house_style",
                7: "building_type",
            }
            plot = px.scatter if idx in {1, 5} else px.box
            chart(
                plot(
                    frame,
                    x=fields[idx],
                    y="sale_price",
                    template=TEMPLATE,
                    color_discrete_sequence=PALETTE,
                )
            )
        show_history(frame)
st.divider()
st.caption(DISCLAIMER)
