# Univariate Analysis — Food HICP Index

## Main series
- **Series:** `food_hicp`
- **Frequency:** monthly
- **Sample:** January 2017 – December 2025
- **Observations:** 108
- **Source:** IMF STA:CPI
- **Code:** `ROU.HICP.CP01.IX.M`

## Descriptive analysis
- The Food HICP index shows a clear upward trend over the full sample.
- The strongest acceleration is visible during the 2021–2023 shock episode.
- The series reaches its maximum at the end of the sample, confirming persistent price growth.
- Descriptive statistics indicate substantial variation over time, consistent with the inflationary episode.

## Deterministic vs. stochastic trend
- The level series was tested using **ADF**, **KPSS_c**, and **KPSS_ct**.
- ADF does **not** reject the null of a unit root in levels.
- KPSS_c rejects level stationarity.
- KPSS_ct also rejects trend stationarity.
- Taken together, these results indicate that the series is not stationary around either a constant or a deterministic trend.
- Therefore, the level series is more consistent with a **stochastic trend**.

## Stationarity
- **Level series:** likely non-stationary
- **Log-level series:** likely non-stationary
- **First-differenced series:** likely stationary
- **First-differenced log series:** likely stationary
- Overall conclusion: the Food HICP series is **integrated of order 1, I(1)**.

## Seasonality
- Seasonality was inspected using:
  - **STL decomposition**
  - **average monthly values**
  - **ACF and PACF plots**
- The monthly pattern suggests recurring within-year variation.
- Average monthly values are lowest in the first part of the year and highest toward the end of the year.
- **December** has the highest average monthly value in the sample.
- STL decomposition also supports the presence of a seasonal component.
- This provides empirical justification for considering a **seasonal forecasting model**.

## Smoothing models
- Three exponential smoothing methods were estimated:
  - **SES**
  - **Holt**
  - **Holt-Winters**
- **SES** is too rigid for a series with strong trend and seasonal variation.
- **Holt** improves on SES by allowing trend.
- **Holt-Winters** is the most appropriate smoothing-based method because it allows both trend and seasonality.

## Holt-Winters interpretation
- Although Holt-Winters is conceptually appropriate for this series, the estimated smoothing parameters show a degenerate behavior:
  - the seasonal smoothing parameter is extremely close to zero
  - the level smoothing parameter is very close to one
- This suggests that the fitted Holt-Winters model is not capturing seasonality in a strong dynamic way.
- Its forecast errors become much larger toward the end of the test period, especially in 2025.
- This helps explain why Holt-Winters performs clearly worse than SARIMA out of sample.

## Final SARIMA model
- The selected model is **SARIMA(1,1,1)(0,1,1,12)**.
- It was selected based on **out-of-sample forecast accuracy** from the candidate grid.
- This model delivers the best test-set performance among the evaluated SARIMA specifications.

## SARIMA diagnostics
- Residual diagnostics show **no strong evidence of remaining autocorrelation** at lag 12 according to the Ljung-Box test.
- However, residual normality is strongly rejected.
- The QQ-plot indicates the presence of large deviations, consistent with the 2021–2022 shock period.
- The seasonal MA parameter is very close to the invertibility boundary and has a very large standard error.
- Therefore, the final SARIMA model should be interpreted mainly through its **forecasting performance**, not through the statistical significance of every individual parameter.

## Final forecast comparison
- **Holt-Winters:** RMSE = 8.8704, MAE = 7.0027, MAPE = 4.0535
- **SARIMA(1,1,1)(0,1,1,12):** RMSE = 2.9288, MAE = 2.2081, MAPE = 1.2808

## Final conclusion
- The Food HICP index is a non-stationary monthly series with a stochastic trend and evidence of seasonality.
- After first differencing, the series becomes stationary, which supports an **I(1)** classification.
- Among the forecasting methods considered, **SARIMA(1,1,1)(0,1,1,12)** clearly outperforms Holt-Winters on the test sample.
- Even though some SARIMA parameters are not cleanly identified and residual normality is rejected, the model is retained because of its clearly superior **out-of-sample forecasting accuracy**.