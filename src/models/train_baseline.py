from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_DATA_PATH = Path("data/processed/GlobalWeatherRepository_features.parquet")
OUTPUT_METRICS_PATH = Path("reports/baseline_metrics.csv")
OUTPUT_PREDICTIONS_PATH = Path("reports/baseline_predictions.parquet")


def load_feature_data(path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Load the engineered feature dataset."""
    return pd.read_parquet(path)


def prepare_baseline_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select the columns required for baseline evaluation and drop rows that cannot be used."""
    df = df.copy()

    required_columns = [
        "location_name",
        "country",
        "last_updated",
        "temperature_celsius",
        "temp_lag_1",
        "temp_lag_2",
        "temp_lag_3",
        "temp_roll_mean_3",
        "temp_roll_std_3",
        "target_temperature_next",
    ]

    df = df[required_columns].dropna().reset_index(drop=True)
    df = df.sort_values(["location_name", "last_updated"]).reset_index(drop=True)
    return df


def temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a simple global time-based split."""
    df = df.sort_values("last_updated").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    denominator = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def build_baseline_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    """Create multiple baseline predictions for comparison."""
    predictions_df = test_df[["location_name", "country", "last_updated", "target_temperature_next"]].copy()

    predictions_df["pred_naive_lag_1"] = test_df["temp_lag_1"]
    predictions_df["pred_naive_lag_2"] = test_df["temp_lag_2"]
    predictions_df["pred_naive_lag_3"] = test_df["temp_lag_3"]
    predictions_df["pred_rolling_mean_3"] = test_df["temp_roll_mean_3"]
    predictions_df["pred_last_observation"] = test_df["temperature_celsius"]

    return predictions_df


def evaluate_baselines(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all baseline strategies and return a ranked metrics table."""
    y_true = predictions_df["target_temperature_next"]

    prediction_columns = {
        "naive_lag_1": "pred_naive_lag_1",
        "naive_lag_2": "pred_naive_lag_2",
        "naive_lag_3": "pred_naive_lag_3",
        "rolling_mean_3": "pred_rolling_mean_3",
        "last_observation": "pred_last_observation",
    }

    results: list[dict[str, float | str]] = []

    for model_name, pred_col in prediction_columns.items():
        metrics = evaluate_predictions(y_true, predictions_df[pred_col])
        results.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
            }
        )

    results_df = pd.DataFrame(results).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return results_df


def save_outputs(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    """Persist baseline evaluation artifacts for reporting."""
    OUTPUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)
    predictions_df.to_parquet(OUTPUT_PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    feature_df = load_feature_data()
    baseline_df = prepare_baseline_dataset(feature_df)
    train_df, test_df = temporal_train_test_split(baseline_df, test_size=0.2)

    baseline_predictions_df = build_baseline_predictions(test_df)
    baseline_metrics_df = evaluate_baselines(baseline_predictions_df)
    save_outputs(baseline_metrics_df, baseline_predictions_df)

    print("Baseline evaluation completed.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("\nBaseline metrics:")
    print(baseline_metrics_df)
    print(f"\nMetrics saved to: {OUTPUT_METRICS_PATH}")
    print(f"Predictions saved to: {OUTPUT_PREDICTIONS_PATH}")