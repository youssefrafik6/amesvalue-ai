# Limitations and responsible use

This project is an educational machine learning demonstration using historical Ames Housing data. It is not a professional appraisal, mortgage valuation, or investment recommendation.

The data are from one city and end in July 2010. Estimates use historical nominal dollars. Market changes, inflation, inspection condition and unrecorded repairs limit current and geographic applicability. Large and unusually priced homes can drive errors and are retained in test. Exponentiated linear models can give extreme predictions for unusual combinations. The chronological model serves the app because its rolling fit/evaluation design reflects forecasting, with its performance still limited to the study period.

Subgroup tables report counts, MAE, median absolute error and interval coverage by neighborhood, training-defined price and size bins, and quality. Groups under 20 test records are flagged. Some categories are absent from individual test sets. Neighborhood can proxy for structural differences in housing and access to resources. The dataset does not support protected-attribute inference or an overall fairness claim. Sparse group coverage and errors should be considered before an automated valuation informs consequential decisions.

The constrained input envelope is broad. Predictions for combinations far from comparable historical sales carry additional uncertainty. Changes in the what-if tool describe model associations. No experiment identifies renovation returns or causal neighborhood premiums.
