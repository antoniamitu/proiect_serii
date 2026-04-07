from pathlib import Path

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


def plot_sarima_residuals(fitted_model) -> Path:
    """
    Plot SARIMA residuals over time.
    """
    ensure_output_directories()

    order = fitted_model.model.order
    seasonal_order = fitted_model.model.seasonal_order
    n_init = order[1] + seasonal_order[1] * seasonal_order[3]
    residuals = fitted_model.resid.iloc[n_init:]

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

    order = fitted_model.model.order
    seasonal_order = fitted_model.model.seasonal_order
    n_init = order[1] + seasonal_order[1] * seasonal_order[3]
    residuals = fitted_model.resid.iloc[n_init:].dropna()

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
    Build a small diagnostics table for SARIMA residuals.
    """
    order = fitted_model.model.order
    seasonal_order = fitted_model.model.seasonal_order
    n_init = order[1] + seasonal_order[1] * seasonal_order[3]
    residuals = fitted_model.resid.iloc[n_init:].dropna()

    lb_df = acorr_ljungbox(residuals, lags=[ljung_box_lags], return_df=True)
    lb_stat = lb_df["lb_stat"].iloc[0]
    lb_pvalue = lb_df["lb_pvalue"].iloc[0]

    norm_stat, norm_pvalue = normaltest(residuals)

    diagnostics_df = pd.DataFrame(
        {
            "metric": [
                "AIC",
                "BIC",
                f"Ljung-Box({ljung_box_lags}) statistic",
                f"Ljung-Box({ljung_box_lags}) p-value",
                "Normality statistic",
                "Normality p-value",
            ],
            "value": [
                fitted_model.aic,
                fitted_model.bic,
                lb_stat,
                lb_pvalue,
                norm_stat,
                norm_pvalue,
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