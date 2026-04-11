from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import (
    SERIES_NAME,
    FIGURES_DIR,
    TABLES_DIR,
    FIG_SIZE,
    DPI,
)


def ensure_output_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def fit_sarima(train: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Fit a SARIMA model on the training series.
    """
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    return fitted_model


def save_sarima_summary_text(fitted_model) -> Path:
    """
    Save SARIMA summary as text.
    """
    ensure_output_directories()
    output_path = TABLES_DIR / "sarima_summary.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fitted_model.summary().as_text())

    return output_path


def generate_sarima_forecast(fitted_model, test: pd.Series) -> pd.DataFrame:
    """
    Generate SARIMA forecasts over the test horizon.
    """
    horizon = len(test)
    forecast_res = fitted_model.get_forecast(steps=horizon)

    conf_int_95 = forecast_res.conf_int(alpha=0.05)
    conf_int_80 = forecast_res.conf_int(alpha=0.20)

    forecast_df = pd.DataFrame(index=test.index)
    forecast_df["actual"] = test.values
    forecast_df["sarima_forecast"] = forecast_res.predicted_mean.values
    forecast_df["lower_95"] = conf_int_95.iloc[:, 0].values
    forecast_df["upper_95"] = conf_int_95.iloc[:, 1].values
    forecast_df["lower_80"] = conf_int_80.iloc[:, 0].values
    forecast_df["upper_80"] = conf_int_80.iloc[:, 1].values

    return forecast_df


def save_sarima_forecast(forecast_df: pd.DataFrame) -> Path:
    """
    Save SARIMA forecast table.
    """
    ensure_output_directories()
    output_path = TABLES_DIR / "sarima_forecast.csv"
    forecast_df.to_csv(output_path)
    return output_path


def plot_sarima_forecast(train: pd.Series, test: pd.Series, forecast_df: pd.DataFrame) -> Path:
    """
    Plot SARIMA forecast with 80% and 95% confidence intervals.
    """
    ensure_output_directories()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(forecast_df.index, forecast_df["sarima_forecast"], label="SARIMA Forecast")

    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_95"],
        forecast_df["upper_95"],
        alpha=0.2,
        label="95% CI",
    )
    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_80"],
        forecast_df["upper_80"],
        alpha=0.3,
        label="80% CI",
    )

    plt.title(f"{SERIES_NAME} - SARIMA(1,1,1)(0,1,1,12) Forecast")
    plt.xlabel("Date")
    plt.ylabel(SERIES_NAME)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "07_sarima_forecast.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def _get_effective_residuals(fitted_model) -> pd.Series:
    """
    Exclude the initial residuals lost to regular and seasonal differencing.
    """
    order = fitted_model.model.order
    seasonal_order = fitted_model.model.seasonal_order
    n_init = order[1] + seasonal_order[1] * seasonal_order[3]
    residuals = fitted_model.resid.iloc[n_init:].dropna()
    return residuals


def plot_sarima_residuals(fitted_model) -> Path:
    """
    Plot SARIMA residuals over time.
    """
    ensure_output_directories()
    residuals = _get_effective_residuals(fitted_model)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(residuals.index, residuals)
    plt.axhline(0, linestyle="--")
    plt.title("SARIMA Residuals")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "08_residuals_sarima.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def plot_sarima_qq(fitted_model) -> Path:
    """
    QQ-plot for SARIMA residuals.
    """
    ensure_output_directories()
    residuals = _get_effective_residuals(fitted_model)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    qqplot(residuals, line="s", ax=ax)
    ax.set_title("QQ-Plot of SARIMA Residuals")
    plt.tight_layout()

    output_path = FIGURES_DIR / "09_qqplot_sarima.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def residual_diagnostics_table(fitted_model, ljung_box_lags=12) -> pd.DataFrame:
    """
    Build an extended diagnostics table for the final SARIMA model.
    Includes residual tests and key parameter diagnostics relevant for interpretation.
    """
    residuals = _get_effective_residuals(fitted_model)

    lb_df = acorr_ljungbox(residuals, lags=[ljung_box_lags], return_df=True)
    lb_stat = lb_df["lb_stat"].iloc[0]
    lb_pvalue = lb_df["lb_pvalue"].iloc[0]

    norm_stat, norm_pvalue = normaltest(residuals)

    param_names = list(fitted_model.param_names)
    params = pd.Series(fitted_model.params, index=param_names)
    bse = pd.Series(fitted_model.bse, index=param_names)
    pvalues = pd.Series(fitted_model.pvalues, index=param_names)

    seasonal_ma_name = next((name for name in param_names if "ma.S" in name), None)

    seasonal_ma_coef = np.nan
    seasonal_ma_std_err = np.nan
    seasonal_ma_pvalue = np.nan
    seasonal_ma_near_boundary = np.nan

    if seasonal_ma_name is not None:
        seasonal_ma_coef = params[seasonal_ma_name]
        seasonal_ma_std_err = bse[seasonal_ma_name]
        seasonal_ma_pvalue = pvalues[seasonal_ma_name]
        seasonal_ma_near_boundary = abs(seasonal_ma_coef) >= 0.98

    diagnostics_df = pd.DataFrame(
        {
            "metric": [
                "AIC",
                "BIC",
                "enforce_stationarity",
                "enforce_invertibility",
                "effective_residual_count",
                f"Ljung-Box({ljung_box_lags}) statistic",
                f"Ljung-Box({ljung_box_lags}) p-value",
                "Normality statistic",
                "Normality p-value",
                "seasonal_ma_parameter",
                "seasonal_ma_coefficient",
                "seasonal_ma_std_error",
                "seasonal_ma_p_value",
                "seasonal_ma_near_boundary",
            ],
            "value": [
                fitted_model.aic,
                fitted_model.bic,
                False,
                False,
                len(residuals),
                lb_stat,
                lb_pvalue,
                norm_stat,
                norm_pvalue,
                seasonal_ma_name,
                seasonal_ma_coef,
                seasonal_ma_std_err,
                seasonal_ma_pvalue,
                seasonal_ma_near_boundary,
            ],
        }
    )

    return diagnostics_df


def save_residual_diagnostics(diagnostics_df: pd.DataFrame) -> Path:
    """
    Save SARIMA residual diagnostics table.
    """
    ensure_output_directories()
    output_path = TABLES_DIR / "sarima_residual_diagnostics.csv"
    diagnostics_df.to_csv(output_path, index=False)
    return output_path


def save_sarima_diagnostic_notes(fitted_model, diagnostics_df: pd.DataFrame) -> Path:
    """
    Save a short interpretation note for the final SARIMA diagnostics.
    """
    ensure_output_directories()
    output_path = TABLES_DIR / "sarima_diagnostic_notes.txt"

    values = dict(zip(diagnostics_df["metric"], diagnostics_df["value"]))

    lb_p = float(values[f"Ljung-Box(12) p-value"])
    norm_p = float(values["Normality p-value"])

    seasonal_name = values["seasonal_ma_parameter"]
    seasonal_coef = values["seasonal_ma_coefficient"]
    seasonal_se = values["seasonal_ma_std_error"]
    seasonal_p = values["seasonal_ma_p_value"]
    seasonal_boundary = values["seasonal_ma_near_boundary"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SARIMA diagnostic notes\n")
        f.write("======================\n\n")

        f.write(
            "Model estimated with enforce_stationarity=False and "
            "enforce_invertibility=False. This allows the optimizer to approach "
            "the boundary of the admissible parameter space if that improves fit.\n\n"
        )

        if lb_p >= 0.05:
            f.write(
                f"Ljung-Box p-value = {lb_p:.4f}: no strong evidence of residual autocorrelation "
                "at the selected lag, so the residuals behave approximately like white noise in this respect.\n\n"
            )
        else:
            f.write(
                f"Ljung-Box p-value = {lb_p:.4f}: residual autocorrelation may still be present.\n\n"
            )

        if norm_p < 0.05:
            f.write(
                f"Normality p-value = {norm_p:.4e}: residual normality is rejected. "
                "This is consistent with the visible outlier / shock period in 2021–2022 and suggests "
                "that the series contains unusually large deviations that a plain SARIMA model cannot fully absorb.\n\n"
            )
        else:
            f.write(
                f"Normality p-value = {norm_p:.4f}: residual normality is not rejected.\n\n"
            )

        if pd.notna(seasonal_coef):
            f.write(
                f"Seasonal MA parameter ({seasonal_name}) = {float(seasonal_coef):.4f}, "
                f"std. error = {float(seasonal_se):.4f}, p-value = {float(seasonal_p):.4f}.\n"
            )

            if str(seasonal_boundary).lower() == "true":
                f.write(
                    "Its absolute value is very close to 1, which suggests that the seasonal MA component is near the boundary "
                    "of invertibility / identification. Therefore, this parameter should not be over-interpreted individually.\n\n"
                )
            else:
                f.write(
                    "This parameter is not near the boundary of invertibility.\n\n"
                )

        f.write(
            "Interpretation rule for the report: the final SARIMA model is retained primarily because of its "
            "strong out-of-sample forecasting performance relative to Holt-Winters, not because every individual "
            "parameter is cleanly identified or statistically significant.\n"
        )

    return output_path