# Original research

**Predicting Ames Housing Sale Prices with Multiple Linear Regression**
Youssef Rafik · MAT 326 · Southern Connecticut State University · Spring 2026

The research asked which property characteristics were associated with sale price after adjustment and how a smaller regression could retain explanatory fit. It also asked whether overall quality had different slopes across neighborhoods. The historical data contain 2,930 Ames, Iowa residential sales measured on 82 fields. The fitted formulas use ten numeric terms and three factors for quality, size, age, remodeling, bathrooms, rooms, location and building type.

## Workflow reproduced in R

The script assigns a one-based original row number before filtering and fits an initial dollar OLS and an initial log OLS to 2,928 complete records. It screens residual spread, linearity, Q–Q shape, Shapiro–Wilk, Durbin–Watson, VIF, leverage, studentized residuals and Cook’s distance. Durbin–Watson depends on row order and cannot establish independence across properties or time. The cleaned raw response is profiled with Box–Cox. Log scaling improved residual behavior, while the final Breusch–Pagan p-value (7.104 × 10⁻¹³) and Q–Q tails show remaining problems for classical inference.

Two records have missing selected fields. The historical script removes 21 sales in levels of size ten or below (Blueste 10, Greens 8, GrnHill 2, Landmrk 1), where coefficients can have high leverage, and three atypically large Edwards sales (1499, 2181, 2182). Their living areas are 5,642, 5,095 and 4,676 sq ft. Exclusion was diagnostic and changes the estimand. The new evaluation retains these sales and uses fold-local imputation.

## Results and source discrepancies

| Historical R model | Response | R² | Adjusted R² | AIC | BIC |
|---|---|---:|---:|---:|---:|
| Initial OLS | Dollars | .8452 | .8426 | 69050.3 | 69349.4 |
| Full cleaned log OLS | Log dollars | .8876 | .8858 | −3213.2 | −2938.4 |
| Stepwise AIC | Log dollars | .8875 | .8858 | −3215.0 | −2952.2 |
| Backward BIC | Log dollars | .8856 | .8843 | −3184.4 | −2969.4 |
| Quality × neighborhood | Log dollars | .8935 | .8909 | −3323.8 | −2911.6 |

Both selection methods move whole terms. AIC starts at the intercept and searches in both directions. BIC starts at the full model and deletes terms. The report calls the full specification “14 predictors” but counts ten numeric and three categorical predictors, matching the 13 source formula terms. The final model keeps quality, living area, basement area, garage area, year built, remodel year, lot area, total rooms, neighborhood, house style and building type. The 43 coefficients include the intercept and treatment-coded indicators. The report calls −3215.0 the best AIC. The interaction has a lower AIC and weaker BIC. The historical choice prioritized simplicity and interpretation. Raw and log likelihoods and R² use different response scales.

Ridge and Lasso reproduce training log R² of .8853 and .8875, with corresponding ten-fold log CV MSE of .01970 and .01946 at the selected penalties. Lasso's one-standard-error choice has 17 nonzero terms. Removing neighborhood reduced the maximum VIF from 18.02 to 5.30 and lowered adjusted log R² to .8711. The reference is Blmngtn. Neighborhood effects are adjusted associations. The five fitted examples reproduce 3.8–10.6% absolute errors and the typical-home example rounds to $206,528 with a $157,393–$271,001 interval. These examples were drawn from the fit data.

The new project evaluates dollar predictions, compares retransformation methods, selects models inside training folds, adds calibration and tests separate records and future sale months. The supplied report remains with the original project and can be linked from a public repository once redistribution is authorized.
