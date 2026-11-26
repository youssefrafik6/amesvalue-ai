# Resume and interview notes

**AmesValue AI** extends an SCSU applied statistics project into a reproducible historical housing study and valuation demo. Stack includes R, Python, pandas, NumPy, scikit-learn, statsmodels, XGBoost, Plotly, Streamlit, FastAPI, Docker, pytest and GitHub Actions.

- Reproduced an R multiple-regression study across 2,930 Ames property sales, verifying a 43-coefficient Stepwise AIC model with adjusted log R² of 0.886.
- Evaluated 16 candidates with parcel-aware and chronological holdouts. The rolling-validation winner achieved $16,567 MAE and 0.894 R² on 497 later sales.
- Built a shared FastAPI and Streamlit prediction engine with conformal intervals covering 90.7% of the chronological holdout at 90% nominal coverage.

| Role | Interview description |
|---|---|
| Data Analyst | Built an interactive housing explorer and compared historical prices with regression-adjusted neighborhood associations. |
| Data Scientist | Reproduced R inference, audited residual assumptions, compared dollar retransformation and measured predictive uncertainty on separate calibration and test sets. |
| ML Engineer | Built reusable scikit-learn pipelines, persisted the model, shared a Pydantic serving contract and tested API and dashboard behavior. |
| Data Engineer | Validated a legacy workbook, preserved source row lineage, published field definitions and generated versioned split and audit outputs. |

## Interview prompts

- **Why log prices?** Right skew and widening residual spread motivate a multiplicative error scale. Exponentiation alone estimates the geometric response; smearing and an approximate lognormal correction were compared in dollars.
- **Why Ridge and Lasso?** Both trade bias for stability in correlated designs. Ridge shrinks terms with an L2 penalty. Lasso uses L1 and can set coefficients to zero.
- **Why does multicollinearity matter?** It can inflate coefficient variance and make separate effects hard to untangle even when joint prediction is useful.
- **Why Stepwise AIC?** Grouped add/drop selection reduced the formula while maintaining historical fit and interpretability. Selection can be unstable and makes unadjusted inference optimistic.
- **Why use a held-out test?** Scoring the same observations used for tuning can conceal generalization failures. One unusually large home exposed instability in the random-split log benchmark.
- **How do prediction intervals work?** The serving interval adds and subtracts a rank-based quantile of absolute residuals computed on a separate calibration split.
- **Why might boosting help?** Tree ensembles can learn nonlinearities and interactions. XGBoost was competitive, while Extra Trees narrowly won rolling CV.
- **What limits the data?** A single city, 2006–2010 nominal prices, sparse neighborhoods and omitted condition details limit geographic and temporal applicability.
- **How was leakage prevented?** Parcel groups and whole calendar months keep repeated units and periods separated. Imputation, scaling, levels and feature selection are learned within training folds.
