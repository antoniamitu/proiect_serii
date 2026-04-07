# Notes - Univariate analysis

## Main series
- food_hicp
- Monthly data: 2017-01 to 2025-12
- 108 observations

## Stationarity
- level: likely non-stationary
- log level: likely non-stationary
- first difference: likely stationary
- conclusion: series is I(1)

## Seasonality
- monthly averages suggest seasonal pattern
- December has the highest monthly average
- STL supports seasonality

## Smoothing models
- SES too rigid
- Holt better than SES
- Holt-Winters suitable due to trend + seasonality

## Final SARIMA
- Selected model: SARIMA(1,1,1)(0,1,1,12)
- chosen based on out-of-sample forecast accuracy

## Final comparison
- Holt-Winters: RMSE 8.8704, MAE 7.0027, MAPE 4.0535
- SARIMA(1,1,1)(0,1,1,12): RMSE 2.9288, MAE 2.2081, MAPE 1.2808