from src.config import SERIES_NAME, SERIES_COLUMN
from src.utils.load_data import (
    load_master_dataset,
    prepare_univariate_series,
    save_processed_series,
)
from src.utils.descriptive import (
    compute_descriptive_statistics,
    save_descriptive_statistics,
    add_log_series,
    plot_series_levels,
    plot_series_log,
)
from src.utils.stationarity import (
    create_stationarity_variants,
    run_stationarity_suite,
    save_stationarity_results,
    build_stationarity_summary,
    save_stationarity_summary,
)
from src.utils.seasonality import (
    plot_stl_decomposition,
    plot_acf_series,
    plot_pacf_series,
    plot_acf_differenced,
    plot_pacf_differenced,
    compute_monthly_seasonality_table,
)
from src.utils.smoothing_models import (
    split_train_test,
    generate_forecasts,
    save_smoothing_forecasts,
    plot_holt_winters_forecast,
    save_smoothing_model_params,
)
from src.utils.sarima_model import (
    fit_sarima,
    save_sarima_summary_text,
    generate_sarima_forecast,
    save_sarima_forecast,
    plot_sarima_forecast,
    plot_sarima_residuals,
    plot_sarima_qq,
    residual_diagnostics_table,
    save_residual_diagnostics,
)
from src.utils.evaluation import (
    compare_models,
    save_model_comparison,
)
from src.utils.sarima_tuning import (
    evaluate_sarima_candidates,
    save_sarima_tuning_results,
)


def main() -> None:
    print("=" * 60)
    print(f"Univariate analysis for: {SERIES_NAME}")
    print("=" * 60)

    df = load_master_dataset()
    print(f"Master dataset loaded successfully. Shape: {df.shape}")

    series_df = prepare_univariate_series(df)
    print("Univariate series prepared successfully.")
    print(series_df.head())
    print()
    print(series_df.tail())
    print()
    print(f"Number of observations: {len(series_df)}")
    print(f"Column used: {SERIES_COLUMN}")
    print(f"Date range: {series_df.index.min().date()} -> {series_df.index.max().date()}")

    processed_path = save_processed_series(series_df)
    print(f"Processed series saved to: {processed_path}")

    stats_df = compute_descriptive_statistics(series_df)
    stats_path = save_descriptive_statistics(stats_df)
    print("\nDescriptive statistics:")
    print(stats_df)
    print(f"\nDescriptive statistics saved to: {stats_path}")

    series_df = add_log_series(series_df)
    print("\nLog-transformed series added successfully.")

    levels_plot_path = plot_series_levels(series_df)
    log_plot_path = plot_series_log(series_df)
    print(f"Levels plot saved to: {levels_plot_path}")
    print(f"Log plot saved to: {log_plot_path}")

    series_df = create_stationarity_variants(series_df)
    print("\nStationarity variants created successfully.")
    print(series_df.head(10))

    stationarity_results_df = run_stationarity_suite(series_df)
    stationarity_results_path = save_stationarity_results(stationarity_results_df)
    print("\nFull stationarity test results:")
    print(stationarity_results_df)
    print(f"\nStationarity results saved to: {stationarity_results_path}")

    stationarity_summary_df = build_stationarity_summary(stationarity_results_df)
    stationarity_summary_path = save_stationarity_summary(stationarity_summary_df)
    print("\nStationarity summary:")
    print(stationarity_summary_df)
    print(f"\nStationarity summary saved to: {stationarity_summary_path}")

    stl_plot_path = plot_stl_decomposition(series_df)
    acf_plot_path = plot_acf_series(series_df)
    pacf_plot_path = plot_pacf_series(series_df)
    acf_diff_plot_path = plot_acf_differenced(series_df)
    pacf_diff_plot_path = plot_pacf_differenced(series_df)
    monthly_seasonality_df = compute_monthly_seasonality_table(series_df)

    print("\nSeasonality analysis completed.")
    print(f"STL decomposition plot saved to: {stl_plot_path}")
    print(f"ACF (level) plot saved to: {acf_plot_path}")
    print(f"PACF (level) plot saved to: {pacf_plot_path}")
    print(f"ACF (differenced) plot saved to: {acf_diff_plot_path}")
    print(f"PACF (differenced) plot saved to: {pacf_diff_plot_path}")
    print("\nAverage monthly values:")
    print(monthly_seasonality_df)

    train, test = split_train_test(series_df)
    print("\nTrain / Test split completed.")
    print(f"Train observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"Train period: {train.index.min().date()} -> {train.index.max().date()}")
    print(f"Test period: {test.index.min().date()} -> {test.index.max().date()}")

    smoothing_forecast_df = generate_forecasts(train, test)
    smoothing_forecasts_path = save_smoothing_forecasts(smoothing_forecast_df)
    hw_plot_path = plot_holt_winters_forecast(train, test, smoothing_forecast_df)
    smoothing_summary_path = save_smoothing_model_params(train)

    print("\nSmoothing models estimated successfully.")
    print(smoothing_forecast_df.head())
    print(f"\nSmoothing forecasts saved to: {smoothing_forecasts_path}")
    print(f"Holt-Winters forecast plot saved to: {hw_plot_path}")
    print(f"Smoothing model summary saved to: {smoothing_summary_path}")

    # SARIMA tuning — rulat primul pentru a justifica alegerea modelului final
    sarima_tuning_df = evaluate_sarima_candidates(train, test)
    sarima_tuning_path = save_sarima_tuning_results(sarima_tuning_df)

    print("\nSARIMA tuning results:")
    print(sarima_tuning_df)
    print(f"\nSARIMA tuning table saved to: {sarima_tuning_path}")

    # SARIMA final — ales pe baza rezultatelor de tuning de mai sus
    final_sarima_order = (1, 1, 1)
    final_sarima_seasonal_order = (0, 1, 1, 12)

    sarima_model = fit_sarima(
        train,
        order=final_sarima_order,
        seasonal_order=final_sarima_seasonal_order,
    )
    sarima_summary_path = save_sarima_summary_text(sarima_model)
    sarima_forecast_df = generate_sarima_forecast(sarima_model, test)
    sarima_forecast_path = save_sarima_forecast(sarima_forecast_df)
    sarima_plot_path = plot_sarima_forecast(train, test, sarima_forecast_df)
    residuals_plot_path = plot_sarima_residuals(sarima_model)
    qq_plot_path = plot_sarima_qq(sarima_model)
    diagnostics_df = residual_diagnostics_table(sarima_model)
    diagnostics_path = save_residual_diagnostics(diagnostics_df)

    print("\nFinal SARIMA model estimated successfully.")
    print(f"Selected model: SARIMA{final_sarima_order}x{final_sarima_seasonal_order}")
    print(sarima_forecast_df.head())
    print(f"\nSARIMA summary saved to: {sarima_summary_path}")
    print(f"SARIMA forecast saved to: {sarima_forecast_path}")
    print(f"SARIMA forecast plot saved to: {sarima_plot_path}")
    print(f"SARIMA residuals plot saved to: {residuals_plot_path}")
    print(f"SARIMA QQ plot saved to: {qq_plot_path}")
    print(f"SARIMA diagnostics saved to: {diagnostics_path}")
    print("\nSARIMA diagnostics:")
    print(diagnostics_df)
    
    # Final comparison
    comparison_df = compare_models(smoothing_forecast_df, sarima_forecast_df)
    comparison_path = save_model_comparison(comparison_df)

    print("\nFinal forecast comparison:")
    print(comparison_df)
    print(f"\nForecast accuracy table saved to: {comparison_path}")

    print("\nFull univariate workflow completed successfully.")


if __name__ == "__main__":
    main()