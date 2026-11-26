# Predictive methodology

The predictive study keeps the original 13 terms for every candidate. Numerical medians, scaling, categorical modes and dummy encodings are fitted on each training fold. Full dummy encoding is used for trees, while linear models use treatment coding. Unknown levels in evaluation encode safely as zeros in the categorical block. Sale year, sale month, parcel and original row organize the split and cannot enter the feature matrix.

Python's grouped stepwise selection minimizes n log(RSS/n) plus either 2 × rank or log(n) × rank. R includes additive likelihood constants when printing AIC and BIC. Python's penalized regressions tune dollar MAE and use different objective normalization from glmnet's log MSE. Inference uses statsmodels with HC3 covariance. Post-selection p-values and intervals are descriptive. A unit change with coefficient β multiplies conditional geometric mean price by exp(β), or changes it by 100 × (exp(β) − 1) percent, holding other terms fixed.

Exponentiating a fitted conditional log mean estimates a geometric mean. Jensen's inequality explains why arithmetic mean prices can be higher. This study compares naive exponentiation, Duan smearing (the training mean of exp(log residual)) and the homoskedastic normal approximation exp(σ²/2). A global correction can miss nonconstant residual distributions. The test results report dollar metrics for all three approaches.

Local explanations replace one input at a time with the training center or modal category. Permutation importance perturbs original variables on calibration records. PDP and ICE change a variable over a sampled quantile grid. Correlated variables may share importance, and perturbations can move off the observed joint distribution.

The serving interval uses finite-sample split conformal absolute residuals. Classical OLS intervals are exponentiated prediction bounds for the training Stepwise AIC benchmark. The residual bootstrap resamples additive calibration residuals and approximates equal-tail bounds around the winner. A separate quantile model was deferred because this small study compares three interval constructions already. Chronological coverage is an empirical backtest because exchangeability can fail under drift.

CatBoost was deferred because this experiment already compares four tree ensembles over the small, low-cardinality feature set. The two external holdouts are the final performance checks for small training-fold searches. An additional nested CV layer was deferred.
