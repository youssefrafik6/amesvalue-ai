# Youssef Rafik, MAT 326, Southern Connecticut State University, Spring 2026.
# Run from repository root with Rscript research/original_R_analysis.R [workbook] [output].
args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args)) args[[1]] else "data/raw/AmesHousing.xls"
output <- if (length(args) >= 2) args[[2]] else "results/original_reproduction"
dir.create(output, recursive = TRUE, showWarnings = FALSE)
library(readxl)
library(corrplot)
library(faraway)
library(lmtest)
library(MASS)
library(glmnet)
pdf(file.path(output, "figures.pdf"), width = 15, height = 8)
ames = read_excel(input)
ames = as.data.frame(ames)

dim(ames)
colnames(ames)

# Build Analysis Dataset and Rename Variables
df = data.frame(
  Original_Row = 1:nrow(ames),
  Sale_Price = ames$SalePrice,
  Overall_Quality = ames$`Overall Qual`,
  Living_Area_SqFt = ames$`Gr Liv Area`,
  Basement_SqFt = ames$`Total Bsmt SF`,
  Garage_SqFt = ames$`Garage Area`,
  Year_Built = ames$`Year Built`,
  Year_Remodeled = ames$`Year Remod/Add`,
  First_Floor_SqFt = ames$`1st Flr SF`,
  Lot_Area_SqFt = ames$`Lot Area`,
  Full_Baths = ames$`Full Bath`,
  Total_Rooms = ames$`TotRms AbvGrd`,
  Neighborhood = ames$Neighborhood,
  House_Style = ames$`House Style`,
  Building_Type = ames$`Bldg Type`
)

# Remove rows with missing values
df = df[complete.cases(df), ]

# Convert categorical variables to factors
df$Neighborhood = as.factor(df$Neighborhood)
df$House_Style = as.factor(df$House_Style)
df$Building_Type = as.factor(df$Building_Type)

dim(df)
summary(df)


# Exploratory Data Analysis
num_vars = c("Sale_Price", "Overall_Quality", "Living_Area_SqFt",
             "Basement_SqFt", "Garage_SqFt", "Year_Built",
             "Year_Remodeled", "First_Floor_SqFt", "Lot_Area_SqFt",
             "Full_Baths", "Total_Rooms")

summary(df[, num_vars])
apply(df[, num_vars], 2, mean)
apply(df[, num_vars], 2, sd)

# Sale price distribution
par(mfrow = c(1, 2))

hist(df$Sale_Price,
     breaks = 40,
     col = 4,
     main = "Histogram of Sale Price",
     xlab = "Sale Price ($)")

boxplot(df$Sale_Price,
        horizontal = TRUE,
        col = 4,
        main = "Boxplot of Sale Price",
        xlab = "Sale Price ($)")

# Check skewness of Sale Price
skew_sale = mean((df$Sale_Price - mean(df$Sale_Price))^3) / sd(df$Sale_Price)^3
skew_sale


# Scatterplots: Numerical Predictors vs Sale Price
par(mfrow = c(2, 5), mar = c(4, 4, 2, 1))

plot(Sale_Price ~ Overall_Quality, data = df, col = 4, pch = 16,
     main = "Sale Price vs Quality",
     xlab = "Overall Quality", ylab = "Sale Price")

plot(Sale_Price ~ Living_Area_SqFt, data = df, col = 4, pch = 16,
     main = "Sale Price vs Living Area",
     xlab = "Living Area SqFt", ylab = "Sale Price")

plot(Sale_Price ~ Basement_SqFt, data = df, col = 4, pch = 16,
     main = "Sale Price vs Basement",
     xlab = "Basement SqFt", ylab = "Sale Price")

plot(Sale_Price ~ Garage_SqFt, data = df, col = 4, pch = 16,
     main = "Sale Price vs Garage",
     xlab = "Garage SqFt", ylab = "Sale Price")

plot(Sale_Price ~ Year_Built, data = df, col = 4, pch = 16,
     main = "Sale Price vs Year Built",
     xlab = "Year Built", ylab = "Sale Price")

plot(Sale_Price ~ Year_Remodeled, data = df, col = 4, pch = 16,
     main = "Sale Price vs Remodel Year",
     xlab = "Year Remodeled", ylab = "Sale Price")

plot(Sale_Price ~ First_Floor_SqFt, data = df, col = 4, pch = 16,
     main = "Sale Price vs First Floor",
     xlab = "First Floor SqFt", ylab = "Sale Price")

plot(Sale_Price ~ Lot_Area_SqFt, data = df, col = 4, pch = 16,
     main = "Sale Price vs Lot Area",
     xlab = "Lot Area SqFt", ylab = "Sale Price")

plot(Sale_Price ~ Full_Baths, data = df, col = 4, pch = 16,
     main = "Sale Price vs Full Baths",
     xlab = "Full Baths", ylab = "Sale Price")

plot(Sale_Price ~ Total_Rooms, data = df, col = 4, pch = 16,
     main = "Sale Price vs Rooms",
     xlab = "Total Rooms", ylab = "Sale Price")


# Correlation Matrix for Numerical Predictors
num_pred = c("Overall_Quality", "Living_Area_SqFt",
             "Basement_SqFt", "Garage_SqFt", "Year_Built",
             "Year_Remodeled", "First_Floor_SqFt",
             "Lot_Area_SqFt", "Full_Baths", "Total_Rooms")

M = cor(df[, num_pred])
round(M, 2)

par(mfrow = c(1, 1))
corrplot(M,
         method = "number",
         type = "upper",
         tl.col = "black",
         tl.cex = 0.8,
         number.cex = 0.7)


# Boxplots for Categorical Predictors
par(mfrow = c(1, 1), mar = c(4, 8, 2, 1))

boxplot(Sale_Price ~ Neighborhood,
        data = df,
        horizontal = TRUE,
        col = 4,
        las = 1,
        main = "Sale Price by Neighborhood",
        xlab = "Sale Price ($)",
        ylab = "")

par(mfrow = c(1, 2), mar = c(4, 7, 2, 1))

boxplot(Sale_Price ~ House_Style,
        data = df,
        horizontal = TRUE,
        col = 6,
        las = 1,
        main = "Sale Price by House Style",
        xlab = "Sale Price ($)",
        ylab = "")

boxplot(Sale_Price ~ Building_Type,
        data = df,
        horizontal = TRUE,
        col = 7,
        las = 1,
        main = "Sale Price by Building Type",
        xlab = "Sale Price ($)",
        ylab = "")

# Sample size by neighborhood
table(df$Neighborhood)



# Initial Multiple Linear Regression Model
mdl_initial = lm(Sale_Price ~ Overall_Quality + Living_Area_SqFt +
                   Basement_SqFt + Garage_SqFt + Year_Built +
                   Year_Remodeled + First_Floor_SqFt + Lot_Area_SqFt +
                   Full_Baths + Total_Rooms +
                   Neighborhood + House_Style + Building_Type,
                 data = df)

summary(mdl_initial)

# Main fit statistics
summary(mdl_initial)$r.squared
summary(mdl_initial)$adj.r.squared
summary(mdl_initial)$sigma
AIC(mdl_initial)
BIC(mdl_initial)

# Overall ANOVA table
anova(mdl_initial)


# Initial Regression Diagnostics
res_initial = residuals(mdl_initial)
fit_initial = fitted(mdl_initial)

n_initial = nrow(df)
k_initial = length(coef(mdl_initial)) - 1

# Residuals vs fitted
par(mfrow = c(1, 1))

plot(fit_initial, res_initial,
     col = 4,
     pch = 16,
     main = "Initial Model: Residuals vs Fitted",
     xlab = "Fitted Sale Price",
     ylab = "Residuals")

abline(h = 0, col = 2, lwd = 2)

# QQ plot and residual histogram
par(mfrow = c(1, 2))

qqnorm(res_initial,
       main = "Initial Model: QQ Plot",
       col = 4,
       pch = 16)

qqline(res_initial, col = 2, lwd = 2)

hist(res_initial,
     breaks = 40,
     col = 4,
     main = "Initial Model: Residual Histogram",
     xlab = "Residuals")

# Standard diagnostic plots
par(mfrow = c(2, 2))
plot(mdl_initial)

# Shapiro-Wilk normality test
shapiro.test(res_initial)

# Durbin-Watson test
dwtest(mdl_initial)

# VIF check
X_initial = model.matrix(mdl_initial)[, -1]
faraway::vif(X_initial)

# Leverage
lev_initial = hatvalues(mdl_initial)
lev_cut_initial = 2 * (k_initial + 1) / n_initial

par(mfrow = c(1, 1))

plot(lev_initial,
     ylab = "Leverage",
     main = "Initial Model: Leverage Plot",
     col = 4,
     pch = 17)

abline(h = lev_cut_initial, col = 2, lwd = 2)

sum(lev_initial > lev_cut_initial, na.rm = TRUE)

# Studentized residuals
stud_initial = rstudent(mdl_initial)

plot(stud_initial,
     ylab = "Studentized Residuals",
     main = "Initial Model: Studentized Residuals",
     col = 6,
     pch = 16)

abline(h = 2, col = 2, lty = 2)
abline(h = -2, col = 2, lty = 2)

# Bonferroni cutoff for possible outliers
bon_initial = qt(0.025 / n_initial,
                 df = n_initial - k_initial - 1,
                 lower.tail = FALSE)

bon_initial

# Fixed with na.rm = TRUE
sum(abs(stud_initial) > bon_initial, na.rm = TRUE)

# Cook's distance
cook_initial = cooks.distance(mdl_initial)

plot(cook_initial,
     ylab = "Cook's Distance",
     main = "Initial Model: Cook's Distance",
     col = 4,
     pch = 17)

abline(h = 4 / n_initial, col = 2, lwd = 2)

# Fixed with na.rm = TRUE
sum(cook_initial > 4 / n_initial, na.rm = TRUE)

# Top influential observations
top_initial = order(cook_initial, decreasing = TRUE)[1:10]

df[top_initial, c("Original_Row", "Sale_Price", "Overall_Quality",
                  "Living_Area_SqFt", "Neighborhood")]

cook_initial[top_initial]
lev_initial[top_initial]
stud_initial[top_initial]

# Log Model on Uncleaned Data

df$Log_Sale_Price = log(df$Sale_Price)

mdl_log_uncleaned = lm(Log_Sale_Price ~ Overall_Quality + Living_Area_SqFt +
                         Basement_SqFt + Garage_SqFt + Year_Built +
                         Year_Remodeled + First_Floor_SqFt + Lot_Area_SqFt +
                         Full_Baths + Total_Rooms +
                         Neighborhood + House_Style + Building_Type,
                       data = df)

summary(mdl_log_uncleaned)

# Main fit statistics
summary(mdl_log_uncleaned)$r.squared
summary(mdl_log_uncleaned)$adj.r.squared
summary(mdl_log_uncleaned)$sigma
AIC(mdl_log_uncleaned)
BIC(mdl_log_uncleaned)


# Diagnostics for Log Model on Uncleaned Data
res_log_unc = residuals(mdl_log_uncleaned)
fit_log_unc = fitted(mdl_log_uncleaned)

n_log_unc = nrow(df)
k_log_unc = length(coef(mdl_log_uncleaned)) - 1

# Residuals vs fitted
par(mfrow = c(1, 1))

plot(fit_log_unc, res_log_unc,
     col = 4,
     pch = 16,
     main = "Log Model (Uncleaned): Residuals vs Fitted",
     xlab = "Fitted Log(Sale Price)",
     ylab = "Residuals")

abline(h = 0, col = 2, lwd = 2)

# QQ plot and histogram
par(mfrow = c(1, 2))

qqnorm(res_log_unc,
       main = "Log Model (Uncleaned): QQ Plot",
       col = 4,
       pch = 16)

qqline(res_log_unc, col = 2, lwd = 2)

hist(res_log_unc,
     breaks = 40,
     col = 4,
     main = "Log Model (Uncleaned): Residual Histogram",
     xlab = "Residuals")

# Standard diagnostic plots
par(mfrow = c(2, 2))
plot(mdl_log_uncleaned)

# Shapiro-Wilk test
shapiro.test(res_log_unc)

# Durbin-Watson test
dwtest(mdl_log_uncleaned)

# VIF
X_log_unc = model.matrix(mdl_log_uncleaned)[, -1]
faraway::vif(X_log_unc)

# Leverage
lev_log_unc = hatvalues(mdl_log_uncleaned)
lev_cut_log_unc = 2 * (k_log_unc + 1) / n_log_unc

par(mfrow = c(1, 1))

plot(lev_log_unc,
     ylab = "Leverage",
     main = "Log Model (Uncleaned): Leverage Plot",
     col = 4,
     pch = 17)

abline(h = lev_cut_log_unc, col = 2, lwd = 2)

sum(lev_log_unc > lev_cut_log_unc, na.rm = TRUE)

# Studentized residuals
stud_log_unc = rstudent(mdl_log_uncleaned)

plot(stud_log_unc,
     ylab = "Studentized Residuals",
     main = "Log Model (Uncleaned): Studentized Residuals",
     col = 6,
     pch = 16)

abline(h = 2, col = 2, lty = 2)
abline(h = -2, col = 2, lty = 2)

bon_log_unc = qt(0.025 / n_log_unc,
                 df = n_log_unc - k_log_unc - 1,
                 lower.tail = FALSE)

bon_log_unc

sum(abs(stud_log_unc) > bon_log_unc, na.rm = TRUE)

# Cook's distance
cook_log_unc = cooks.distance(mdl_log_uncleaned)

plot(cook_log_unc,
     ylab = "Cook's Distance",
     main = "Log Model (Uncleaned): Cook's Distance",
     col = 4,
     pch = 17)

abline(h = 4 / n_log_unc, col = 2, lwd = 2)

sum(cook_log_unc > 4 / n_log_unc, na.rm = TRUE)

top_log_unc = order(cook_log_unc, decreasing = TRUE)[1:10]

df[top_log_unc, c("Original_Row", "Sale_Price", "Overall_Quality",
                  "Living_Area_SqFt", "Neighborhood")]

cook_log_unc[top_log_unc]
lev_log_unc[top_log_unc]
stud_log_unc[top_log_unc]
# Data Cleaning

table(df$Neighborhood)

# Remove neighborhoods with very small sample sizes
# These groups have very few observations, so they can create leverage problems.
rare_neighborhoods = names(table(df$Neighborhood))[table(df$Neighborhood) <= 10]
rare_neighborhoods

df_clean = df[!(df$Neighborhood %in% rare_neighborhoods), ]

# Remove known influential observations from the initial model.
# These rows were large homes with unusual sale prices and strong influence.
outlier_rows = c(1499, 2181, 2182)

df_clean = df_clean[!(df_clean$Original_Row %in% outlier_rows), ]

# Drop unused factor levels after removing rows
df_clean$Neighborhood = droplevels(df_clean$Neighborhood)
df_clean$House_Style = droplevels(df_clean$House_Style)
df_clean$Building_Type = droplevels(df_clean$Building_Type)

dim(df)
dim(df_clean)

table(df_clean$Neighborhood)


# Box-Cox Transformation Check
mdl_clean_raw = lm(Sale_Price ~ Overall_Quality + Living_Area_SqFt +
                     Basement_SqFt + Garage_SqFt + Year_Built +
                     Year_Remodeled + First_Floor_SqFt + Lot_Area_SqFt +
                     Full_Baths + Total_Rooms +
                     Neighborhood + House_Style + Building_Type,
                   data = df_clean)

par(mfrow = c(1, 1))

boxcox(mdl_clean_raw,
       lambda = seq(-2, 2, 0.1),
       main = "Box-Cox Plot for Sale Price")

# The Box-Cox plot usually supports a transformation close to lambda = 0.
# Lambda = 0 corresponds to using log(Sale_Price).

df_clean$Log_Sale_Price = log(df_clean$Sale_Price)


# Log-Response Multiple Regression Model
mdl_log_full = lm(Log_Sale_Price ~ Overall_Quality + Living_Area_SqFt +
                    Basement_SqFt + Garage_SqFt + Year_Built +
                    Year_Remodeled + First_Floor_SqFt + Lot_Area_SqFt +
                    Full_Baths + Total_Rooms +
                    Neighborhood + House_Style + Building_Type,
                  data = df_clean)

summary(mdl_log_full)

summary(mdl_log_full)$r.squared
summary(mdl_log_full)$adj.r.squared
summary(mdl_log_full)$sigma
AIC(mdl_log_full)
BIC(mdl_log_full)


# Diagnostics for Log-Response Model
res_log = residuals(mdl_log_full)
fit_log = fitted(mdl_log_full)

n_log = nrow(df_clean)
k_log = length(coef(mdl_log_full)) - 1

# Residuals vs fitted
par(mfrow = c(1, 1))

plot(fit_log, res_log,
     col = 4,
     pch = 16,
     main = "Log Model: Residuals vs Fitted",
     xlab = "Fitted Log(Sale Price)",
     ylab = "Residuals")

abline(h = 0, col = 2, lwd = 2)

# QQ plot and histogram
par(mfrow = c(1, 2))

qqnorm(res_log,
       main = "Log Model: QQ Plot",
       col = 4,
       pch = 16)

qqline(res_log, col = 2, lwd = 2)

hist(res_log,
     breaks = 40,
     col = 4,
     main = "Log Model: Residual Histogram",
     xlab = "Residuals")

# Standard diagnostic plots
par(mfrow = c(2, 2))
plot(mdl_log_full)

# Shapiro-Wilk test
shapiro.test(res_log)

# Durbin-Watson test
dwtest(mdl_log_full)

# VIF
X_log = model.matrix(mdl_log_full)[, -1]
faraway::vif(X_log)

# Leverage
lev_log = hatvalues(mdl_log_full)
lev_cut_log = 2 * (k_log + 1) / n_log

par(mfrow = c(1, 1))

plot(lev_log,
     ylab = "Leverage",
     main = "Log Model: Leverage Plot",
     col = 4,
     pch = 17)

abline(h = lev_cut_log, col = 2, lwd = 2)

sum(lev_log > lev_cut_log, na.rm = TRUE)

# Studentized residuals
stud_log = rstudent(mdl_log_full)

plot(stud_log,
     ylab = "Studentized Residuals",
     main = "Log Model: Studentized Residuals",
     col = 6,
     pch = 16)

abline(h = 2, col = 2, lty = 2)
abline(h = -2, col = 2, lty = 2)

bon_log = qt(0.025 / n_log,
             df = n_log - k_log - 1,
             lower.tail = FALSE)

bon_log

sum(abs(stud_log) > bon_log, na.rm = TRUE)

# Cook's distance
cook_log = cooks.distance(mdl_log_full)

plot(cook_log,
     ylab = "Cook's Distance",
     main = "Log Model: Cook's Distance",
     col = 4,
     pch = 17)

abline(h = 4 / n_log, col = 2, lwd = 2)

sum(cook_log > 4 / n_log, na.rm = TRUE)

top_log = order(cook_log, decreasing = TRUE)[1:10]

df_clean[top_log, c("Original_Row", "Sale_Price", "Overall_Quality",
                    "Living_Area_SqFt", "Neighborhood")]

cook_log[top_log]
lev_log[top_log]
stud_log[top_log]


# Stepwise Selection

# Full and null models for stepwise selection
mdl_null = lm(Log_Sale_Price ~ 1, data = df_clean)

# Stepwise selection using AIC
step_AIC = step(mdl_null,
                scope = list(lower = mdl_null, upper = mdl_log_full),
                direction = "both",
                trace = FALSE)

summary(step_AIC)
AIC(step_AIC)
BIC(step_AIC)

# Backward selection using BIC
step_BIC = step(mdl_log_full,
                direction = "backward",
                k = log(nrow(df_clean)),
                trace = FALSE)

summary(step_BIC)
AIC(step_BIC)
BIC(step_BIC)

# Compare full log model, AIC stepwise, and BIC stepwise
step_compare = data.frame(
  Model = c("Full Log Model", "Stepwise AIC", "Backward BIC"),
  R_squared = c(summary(mdl_log_full)$r.squared,
                summary(step_AIC)$r.squared,
                summary(step_BIC)$r.squared),
  Adjusted_R_squared = c(summary(mdl_log_full)$adj.r.squared,
                         summary(step_AIC)$adj.r.squared,
                         summary(step_BIC)$adj.r.squared),
  AIC = c(AIC(mdl_log_full), AIC(step_AIC), AIC(step_BIC)),
  BIC = c(BIC(mdl_log_full), BIC(step_AIC), BIC(step_BIC))
)

step_compare



# Ridge and Lasso Regression

# Ridge and Lasso use a design matrix.
X = model.matrix(Log_Sale_Price ~ Overall_Quality + Living_Area_SqFt +
                   Basement_SqFt + Garage_SqFt + Year_Built +
                   Year_Remodeled + First_Floor_SqFt + Lot_Area_SqFt +
                   Full_Baths + Total_Rooms +
                   Neighborhood + House_Style + Building_Type,
                 data = df_clean)[, -1]

y = df_clean$Log_Sale_Price

set.seed(123)

# Ridge regression: alpha = 0
cv_ridge = cv.glmnet(X, y, alpha = 0)

plot(cv_ridge, main = "Ridge Cross-Validation")

cv_ridge$lambda.min
cv_ridge$lambda.1se

ridge_coef_min = coef(cv_ridge, s = "lambda.min")
head(as.matrix(ridge_coef_min), 20)

# Lasso regression: alpha = 1
cv_lasso = cv.glmnet(X, y, alpha = 1)

plot(cv_lasso, main = "Lasso Cross-Validation")

cv_lasso$lambda.min
cv_lasso$lambda.1se

lasso_coef_min = coef(cv_lasso, s = "lambda.min")
lasso_coef_1se = coef(cv_lasso, s = "lambda.1se")

# Variables kept by Lasso using lambda.1se
lasso_kept = rownames(lasso_coef_1se)[which(lasso_coef_1se != 0)]
lasso_kept

# Number of predictors kept, excluding intercept
sum(lasso_coef_1se != 0) - 1

# Coefficient path plots
ridge_full = glmnet(X, y, alpha = 0)
lasso_full = glmnet(X, y, alpha = 1)

par(mfrow = c(1, 2))

plot(ridge_full,
     xvar = "lambda",
     main = "Ridge Coefficient Path")

plot(lasso_full,
     xvar = "lambda",
     main = "Lasso Coefficient Path")

# In-sample R-squared for ridge and lasso at lambda.min
pred_ridge = predict(cv_ridge, newx = X, s = "lambda.min")
pred_lasso = predict(cv_lasso, newx = X, s = "lambda.min")

ridge_R2 = 1 - sum((y - pred_ridge)^2) / sum((y - mean(y))^2)
lasso_R2 = 1 - sum((y - pred_lasso)^2) / sum((y - mean(y))^2)

ridge_R2
lasso_R2

# Cross-validated MSE at lambda.min
ridge_CV_MSE = cv_ridge$cvm[which(cv_ridge$lambda == cv_ridge$lambda.min)]
lasso_CV_MSE = cv_lasso$cvm[which(cv_lasso$lambda == cv_lasso$lambda.min)]

ridge_CV_MSE
lasso_CV_MSE


# Interaction Model
mdl_interaction = lm(Log_Sale_Price ~ Overall_Quality + Living_Area_SqFt +
                       Basement_SqFt + Garage_SqFt + Year_Built +
                       Year_Remodeled + First_Floor_SqFt + Lot_Area_SqFt +
                       Full_Baths + Total_Rooms +
                       Neighborhood + House_Style + Building_Type +
                       Overall_Quality:Neighborhood,
                     data = df_clean)

summary(mdl_interaction)

summary(mdl_interaction)$r.squared
summary(mdl_interaction)$adj.r.squared
AIC(mdl_interaction)
BIC(mdl_interaction)

# Compare interaction model to full log model
anova(mdl_log_full, mdl_interaction)


# Diagnostics for Interaction Model
res_int = residuals(mdl_interaction)
fit_int = fitted(mdl_interaction)

n_int = nrow(df_clean)
k_int = length(coef(mdl_interaction)) - 1

par(mfrow = c(1, 1))

plot(fit_int, res_int,
     col = 4,
     pch = 16,
     main = "Interaction Model: Residuals vs Fitted",
     xlab = "Fitted Log(Sale Price)",
     ylab = "Residuals")

abline(h = 0, col = 2, lwd = 2)

par(mfrow = c(1, 2))

qqnorm(res_int,
       main = "Interaction Model: QQ Plot",
       col = 4,
       pch = 16)

qqline(res_int, col = 2, lwd = 2)

hist(res_int,
     breaks = 40,
     col = 4,
     main = "Interaction Model: Residual Histogram",
     xlab = "Residuals")

par(mfrow = c(2, 2))
plot(mdl_interaction)

# Formal tests
shapiro.test(res_int)
dwtest(mdl_interaction)

# Influence checks
cook_int = cooks.distance(mdl_interaction)
stud_int = rstudent(mdl_interaction)
lev_int = hatvalues(mdl_interaction)

bon_int = qt(0.025 / n_int,
             df = n_int - k_int - 1,
             lower.tail = FALSE)

sum(abs(stud_int) > bon_int, na.rm = TRUE)
sum(cook_int > 4 / n_int, na.rm = TRUE)


# Final Model Comparison
final_compare = data.frame(
  Model = c("Initial Raw Sale Price Model",
            "Full Log Model",
            "Stepwise AIC Log Model",
            "Backward BIC Log Model",
            "Ridge Log Model",
            "Lasso Log Model",
            "Log Model with Quality x Neighborhood Interaction"),

  Response = c("Sale_Price",
               "Log_Sale_Price",
               "Log_Sale_Price",
               "Log_Sale_Price",
               "Log_Sale_Price",
               "Log_Sale_Price",
               "Log_Sale_Price"),

  R_squared = c(summary(mdl_initial)$r.squared,
                summary(mdl_log_full)$r.squared,
                summary(step_AIC)$r.squared,
                summary(step_BIC)$r.squared,
                ridge_R2,
                lasso_R2,
                summary(mdl_interaction)$r.squared),

  Adjusted_R_squared = c(summary(mdl_initial)$adj.r.squared,
                         summary(mdl_log_full)$adj.r.squared,
                         summary(step_AIC)$adj.r.squared,
                         summary(step_BIC)$adj.r.squared,
                         NA,
                         NA,
                         summary(mdl_interaction)$adj.r.squared),

  AIC = c(AIC(mdl_initial),
          AIC(mdl_log_full),
          AIC(step_AIC),
          AIC(step_BIC),
          NA,
          NA,
          AIC(mdl_interaction)),

  BIC = c(BIC(mdl_initial),
          BIC(mdl_log_full),
          BIC(step_AIC),
          BIC(step_BIC),
          NA,
          NA,
          BIC(mdl_interaction)),

  CV_MSE = c(NA,
             NA,
             NA,
             NA,
             ridge_CV_MSE,
             lasso_CV_MSE,
             NA)
)

final_compare


# Final Model Choice

final_model = step_AIC

summary(final_model)
confint(final_model)

# Final diagnostics
par(mfrow = c(2, 2))
plot(final_model)

bptest(final_model)

# Final model VIF
X_final = model.matrix(final_model)[, -1]
faraway::vif(X_final)


# Prediction Example Using the Final Model

# Predict the price of a typical home.
# Since the model predicts log(Sale_Price), we exponentiate the prediction
# to return to the dollar scale.

new_home = data.frame(
  Overall_Quality = 7,
  Living_Area_SqFt = mean(df_clean$Living_Area_SqFt),
  Basement_SqFt = mean(df_clean$Basement_SqFt),
  Garage_SqFt = mean(df_clean$Garage_SqFt),
  Year_Built = 2000,
  Year_Remodeled = 2005,
  First_Floor_SqFt = mean(df_clean$First_Floor_SqFt),
  Lot_Area_SqFt = mean(df_clean$Lot_Area_SqFt),
  Full_Baths = 2,
  Total_Rooms = 7,
  Neighborhood = factor("NAmes", levels = levels(df_clean$Neighborhood)),
  House_Style = factor("1Story", levels = levels(df_clean$House_Style)),
  Building_Type = factor("1Fam", levels = levels(df_clean$Building_Type))
)

# Confidence interval for the mean predicted log price
pred_log_ci = predict(final_model,
                      newdata = new_home,
                      interval = "confidence",
                      level = 0.95)

pred_log_ci

# Prediction interval for an individual home
pred_log_pi = predict(final_model,
                      newdata = new_home,
                      interval = "prediction",
                      level = 0.95)

pred_log_pi

# Convert predictions back to dollars
pred_dollar_ci = exp(pred_log_ci)
pred_dollar_pi = exp(pred_log_pi)

pred_dollar_ci
pred_dollar_pi

# Final Model Equation Support

# Print the final model formula
formula(final_model)

# Print final model coefficients
coef(final_model)

# Coefficient table
final_coef_table = as.data.frame(summary(final_model)$coefficients)

final_coef_table$Variable = rownames(final_coef_table)

colnames(final_coef_table) = c("Estimate",
                               "Std_Error",
                               "t_value",
                               "p_value",
                               "Variable")

# Reorder columns
final_coef_table = final_coef_table[, c("Variable",
                                         "Estimate",
                                         "Std_Error",
                                         "t_value",
                                         "p_value")]

# View coefficient table
final_coef_table

# Save the final formula as text
final_formula_text = paste(deparse(formula(final_model)), collapse = " ")
final_formula_text


# VIF Sensitivity Analysis

# Current final model VIF
X_final = model.matrix(final_model)[, -1]
vif_final = faraway::vif(X_final)

vif_final

# Helper function to get max VIF safely
get_max_vif = function(model){
  X = model.matrix(model)[, -1]
  vif_values = faraway::vif(X)
  return(max(vif_values, na.rm = TRUE))
}

# Helper function to count VIF values above 10
get_vif_over_10 = function(model){
  X = model.matrix(model)[, -1]
  vif_values = faraway::vif(X)
  return(sum(vif_values > 10, na.rm = TRUE))
}

# Helper function for Breusch-Pagan p-value
get_bp_pvalue = function(model){
  return(bptest(model)$p.value)
}


# Reduced VIF Model 1:
# Remove Total_Rooms.
# First_Floor_SqFt was already dropped by Stepwise AIC.
# This checks whether removing Total_Rooms lowers VIF while keeping
# Neighborhood in the model.
reduced_vif_model_1 = lm(Log_Sale_Price ~ Overall_Quality +
                           Living_Area_SqFt +
                           Basement_SqFt +
                           Garage_SqFt +
                           Year_Built +
                           Year_Remodeled +
                           Lot_Area_SqFt +
                           Neighborhood +
                           House_Style +
                           Building_Type,
                         data = df_clean)

summary(reduced_vif_model_1)

X_reduced_1 = model.matrix(reduced_vif_model_1)[, -1]
vif_reduced_1 = faraway::vif(X_reduced_1)

vif_reduced_1



# Reduced VIF Model 2:
# Remove Total_Rooms and Basement_SqFt.
# This checks whether removing another size-related variable lowers
# multicollinearity, and whether the loss in fit is worth it.
reduced_vif_model_2 = lm(Log_Sale_Price ~ Overall_Quality +
                           Living_Area_SqFt +
                           Garage_SqFt +
                           Year_Built +
                           Year_Remodeled +
                           Lot_Area_SqFt +
                           Neighborhood +
                           House_Style +
                           Building_Type,
                         data = df_clean)

summary(reduced_vif_model_2)

X_reduced_2 = model.matrix(reduced_vif_model_2)[, -1]
vif_reduced_2 = faraway::vif(X_reduced_2)

vif_reduced_2


# No Neighborhood Model:
# Remove Neighborhood completely.
# This tests how much predictive/explanatory power is lost when location
# is removed from the model.
no_neighborhood_model = lm(Log_Sale_Price ~ Overall_Quality +
                             Living_Area_SqFt +
                             Basement_SqFt +
                             Garage_SqFt +
                             Year_Built +
                             Year_Remodeled +
                             Lot_Area_SqFt +
                             Total_Rooms +
                             House_Style +
                             Building_Type,
                           data = df_clean)

summary(no_neighborhood_model)

X_no_neighborhood = model.matrix(no_neighborhood_model)[, -1]
vif_no_neighborhood = faraway::vif(X_no_neighborhood)

vif_no_neighborhood


# Compare Final Model vs Reduced VIF Models
vif_compare_table = data.frame(
  Model = c("Final Stepwise AIC Log Model",
            "Reduced VIF Model 1: Remove Total_Rooms",
            "Reduced VIF Model 2: Remove Total_Rooms + Basement_SqFt",
            "No Neighborhood Model"),

  R_squared = c(summary(final_model)$r.squared,
                summary(reduced_vif_model_1)$r.squared,
                summary(reduced_vif_model_2)$r.squared,
                summary(no_neighborhood_model)$r.squared),

  Adjusted_R_squared = c(summary(final_model)$adj.r.squared,
                         summary(reduced_vif_model_1)$adj.r.squared,
                         summary(reduced_vif_model_2)$adj.r.squared,
                         summary(no_neighborhood_model)$adj.r.squared),

  AIC = c(AIC(final_model),
          AIC(reduced_vif_model_1),
          AIC(reduced_vif_model_2),
          AIC(no_neighborhood_model)),

  BIC = c(BIC(final_model),
          BIC(reduced_vif_model_1),
          BIC(reduced_vif_model_2),
          BIC(no_neighborhood_model)),

  Max_VIF = c(get_max_vif(final_model),
              get_max_vif(reduced_vif_model_1),
              get_max_vif(reduced_vif_model_2),
              get_max_vif(no_neighborhood_model)),

  Num_VIF_Over_10 = c(get_vif_over_10(final_model),
                      get_vif_over_10(reduced_vif_model_1),
                      get_vif_over_10(reduced_vif_model_2),
                      get_vif_over_10(no_neighborhood_model)),

  BP_p_value = c(get_bp_pvalue(final_model),
                 get_bp_pvalue(reduced_vif_model_1),
                 get_bp_pvalue(reduced_vif_model_2),
                 get_bp_pvalue(no_neighborhood_model))
)

# Round table for cleaner output
vif_compare_table$R_squared = round(vif_compare_table$R_squared, 4)
vif_compare_table$Adjusted_R_squared = round(vif_compare_table$Adjusted_R_squared, 4)
vif_compare_table$AIC = round(vif_compare_table$AIC, 1)
vif_compare_table$BIC = round(vif_compare_table$BIC, 1)
vif_compare_table$Max_VIF = round(vif_compare_table$Max_VIF, 2)
vif_compare_table$BP_p_value = signif(vif_compare_table$BP_p_value, 4)

vif_compare_table


# Real Sample Prediction Table

# Sort homes by actual sale price
df_ordered = df_clean[order(df_clean$Sale_Price), ]

# Select real homes from different price ranges
low_home = df_ordered[round(0.10 * nrow(df_ordered)), ]
q1_home = df_ordered[round(0.25 * nrow(df_ordered)), ]
middle_home = df_ordered[round(0.50 * nrow(df_ordered)), ]
q3_home = df_ordered[round(0.75 * nrow(df_ordered)), ]
high_home = df_ordered[round(0.90 * nrow(df_ordered)), ]

sample_homes = rbind(low_home, q1_home, middle_home, q3_home, high_home)

# Predict on log scale with prediction intervals
sample_pred_log = predict(final_model,
                          newdata = sample_homes,
                          interval = "prediction",
                          level = 0.95)

# Convert predictions back to dollars
sample_pred_dollars = exp(sample_pred_log)

# Create prediction comparison table
sample_prediction_table = data.frame(
  Original_Row = sample_homes$Original_Row,
  Neighborhood = sample_homes$Neighborhood,
  Overall_Quality = sample_homes$Overall_Quality,
  Living_Area_SqFt = sample_homes$Living_Area_SqFt,
  Actual_Sale_Price = sample_homes$Sale_Price,
  Predicted_Sale_Price = sample_pred_dollars[, "fit"],
  PI_Lower = sample_pred_dollars[, "lwr"],
  PI_Upper = sample_pred_dollars[, "upr"]
)

# Prediction error:
sample_prediction_table$Error = sample_prediction_table$Actual_Sale_Price -
  sample_prediction_table$Predicted_Sale_Price

sample_prediction_table$Absolute_Error = abs(sample_prediction_table$Error)

sample_prediction_table$Percent_Error = 100 *
  sample_prediction_table$Absolute_Error /
  sample_prediction_table$Actual_Sale_Price

# Round values for clean table
sample_prediction_table$Actual_Sale_Price = round(sample_prediction_table$Actual_Sale_Price, 0)
sample_prediction_table$Predicted_Sale_Price = round(sample_prediction_table$Predicted_Sale_Price, 0)
sample_prediction_table$PI_Lower = round(sample_prediction_table$PI_Lower, 0)
sample_prediction_table$PI_Upper = round(sample_prediction_table$PI_Upper, 0)
sample_prediction_table$Error = round(sample_prediction_table$Error, 0)
sample_prediction_table$Absolute_Error = round(sample_prediction_table$Absolute_Error, 0)
sample_prediction_table$Percent_Error = round(sample_prediction_table$Percent_Error, 1)

sample_prediction_table


# Neighborhood Coefficient Interpretation

# Identify reference neighborhood
levels(df_clean$Neighborhood)

reference_neighborhood = levels(df_clean$Neighborhood)[1]
reference_neighborhood

# Extract coefficient table
coef_table = as.data.frame(summary(final_model)$coefficients)
coef_table$Variable = rownames(coef_table)

colnames(coef_table) = c("Estimate",
                         "Std_Error",
                         "t_value",
                         "p_value",
                         "Variable")

# Keep only Neighborhood coefficients
neighborhood_effects = coef_table[grep("^Neighborhood", coef_table$Variable), ]

# Clean neighborhood names
neighborhood_effects$Neighborhood = gsub("Neighborhood", "",
                                         neighborhood_effects$Variable)

# Convert log coefficients to exact percent effects
# Exact percent effect = 100 * (exp(beta) - 1)
neighborhood_effects$Percent_Effect_vs_Reference =
  100 * (exp(neighborhood_effects$Estimate) - 1)

# Add significance label
neighborhood_effects$Significance = ifelse(neighborhood_effects$p_value < 0.001, "***",
                                    ifelse(neighborhood_effects$p_value < 0.01, "**",
                                    ifelse(neighborhood_effects$p_value < 0.05, "*",
                                    ifelse(neighborhood_effects$p_value < 0.10, ".", ""))))

# Keep useful columns
neighborhood_effects = neighborhood_effects[, c("Neighborhood",
                                                 "Estimate",
                                                 "Percent_Effect_vs_Reference",
                                                 "p_value",
                                                 "Significance")]

# Round values
neighborhood_effects$Estimate = round(neighborhood_effects$Estimate, 4)
neighborhood_effects$Percent_Effect_vs_Reference =
  round(neighborhood_effects$Percent_Effect_vs_Reference, 1)
neighborhood_effects$p_value = signif(neighborhood_effects$p_value, 4)

# Sort by percent effect
neighborhood_effects_sorted = neighborhood_effects[
  order(neighborhood_effects$Percent_Effect_vs_Reference,
        decreasing = TRUE), ]

neighborhood_effects_sorted

# Significant neighborhoods only
significant_neighborhoods = neighborhood_effects_sorted[
  neighborhood_effects_sorted$p_value < 0.05, ]

significant_neighborhoods


dev.off()
save_table <- function(x, name, row.names = FALSE) {
  write.csv(x, file.path(output, paste0(name, ".csv")), row.names = row.names)
}
save_table(final_compare, "model_comparison")
save_table(step_compare, "stepwise")
save_table(vif_compare_table, "vif_sensitivity")
save_table(final_coef_table, "coefficients")
save_table(neighborhood_effects_sorted, "neighborhood_effects")
save_table(sample_prediction_table, "sample_predictions")
save_table(exp(pred_log_pi), "typical_home")
save_table(M, "correlation", TRUE)
save_table(as.data.frame(t(sapply(df[num_vars], function(x) {
  c(mean = mean(x), sd = sd(x), min = min(x), median = median(x), max = max(x))
}))), "summary", TRUE)
save_table(data.frame(neighborhood = rare_neighborhoods,
                      count = as.integer(table(df$Neighborhood)[rare_neighborhoods])), "rare")
save_table(data.frame(original_row = df$Original_Row, cook = cook_initial,
                      leverage = lev_initial, studentized = stud_initial), "initial_influence")
save_table(data.frame(raw = nrow(ames), complete = nrow(df), cleaned = nrow(df_clean),
                      coefficients = length(coef(final_model)),
                      terms = paste(attr(terms(final_model), "term.labels"), collapse = ", "),
                      reference = reference_neighborhood), "audit")
save_table(data.frame(ridge_min = cv_ridge$lambda.min, lasso_min = cv_lasso$lambda.min,
                      lasso_1se = cv_lasso$lambda.1se,
                      lasso_nonzero = sum(lasso_coef_1se != 0) - 1), "regularization")
for (name in c("mdl_initial", "mdl_log_full", "final_model")) {
  model <- get(name)
  save_table(data.frame(fitted = fitted(model), residual = residuals(model),
    leverage = hatvalues(model), cook = cooks.distance(model),
    standardized = rstandard(model)), paste0(name, "_diagnostics"))
  save_table(data.frame(vif = faraway::vif(model.matrix(model)[, -1])),
             paste0(name, "_vif"), TRUE)
  save_table(data.frame(bp_pvalue = bptest(model)$p.value), paste0(name, "_tests"))
}
bc <- boxcox(mdl_clean_raw, lambda = seq(-0.5, 0.5, 0.01), plotit = FALSE)
save_table(data.frame(lambda = bc$x, log_likelihood = bc$y), "box_cox")
scale_models <- list("Initial raw OLS" = mdl_initial,
                     "Complete-case log OLS" = mdl_log_uncleaned,
                     "Cleaned log OLS" = mdl_log_full)
scale_compare <- do.call(rbind, lapply(scale_models, function(model) {
  fit <- summary(model)
  data.frame(response = all.vars(formula(model))[[1]], rows = nobs(model),
             r_squared = fit$r.squared, adjusted_r_squared = fit$adj.r.squared,
             residual_standard_error = fit$sigma, aic = AIC(model), bic = BIC(model),
             shapiro_w = unname(shapiro.test(residuals(model))$statistic))
}))
save_table(scale_compare, "response_scale_comparison", TRUE)
