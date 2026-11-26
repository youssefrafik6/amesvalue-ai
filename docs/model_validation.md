# Model validation

| Split / CV-selected model | Train | Calibration | Test | CV MAE | Test MAE | Test RMSE | Dollar R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random / Stepwise AIC | 1,875 | 469 | 586 | $16,847 | $21,671 | $100,480 | −0.397 |
| Chronological / Extra Trees | 2,080 | 353 | 497 | $18,977 | $16,567 | $24,417 | 0.894 |

Seed 326 controls random partitioning, calibration partitioning, resampling and randomized model fits. The random split reserves 20% for test, then reserves 20% of development for calibration. Four grouped validation folds select by dollar MAE. The chronological split trains through April 2009, calibrates May–August 2009, and tests September 2009–July 2010. Four expanding-window folds validate on later months. The searches use small fixed grids and an independent external test set. CV scores reflect tuning and can be optimistic. The two protocols reuse parts of one dataset and are dependent analyses.

All 16 candidates use identical test rows within each protocol. The random CV-selected model extrapolated to about $2.17M for original row 1499, a 5,642 ft² Edwards sale recorded at $160,000. That single error contributes roughly 69% of random test squared error for Stepwise AIC. The record remains in the reported test set. Removing the three largest finite Cook’s distances inside each training split is a calibration sensitivity check. In chronological calibration, Stepwise AIC MAE moved from $16,686 to $16,182 under this experiment. The deployed model uses the full chronological training partition.

The serving decision favors the rolling chronological workflow because the intended simulation is a later sale. The protocol decision was made during the results audit. Within each protocol the winner was chosen using training folds before its test was scored. Chronological XGBoost had slightly higher CV MAE ($18,987) and lower test MAE ($15,838), illustrating the sampling uncertainty of the ranking. Extra Trees’ median error is $11,401 and median absolute percentage error is 6.87%. The chronological median baseline MAE is $54,042.

| Chronological interval method | Nominal | Observed | Mean width |
|---|---:|---:|---:|
| Split conformal around Extra Trees | 90% | 90.7% | $78,728 |
| Calibration residual bootstrap around Extra Trees | 90% | 90.9% | $79,338 |
| Classical prediction bounds from Stepwise AIC | 90% | 92.4% | $87,231 |

Subgroup metrics and interval widths are measured on held-out records. Marginal coverage can conceal differences across subgroups. The prospective meaning of chronological coverage depends on future distributions.

[Machine-readable metrics](../results/model_comparison.csv) · [Random subgroup analysis](../results/random_subgroups.csv) · [Chronological subgroup analysis](../results/chronological_subgroups.csv)
