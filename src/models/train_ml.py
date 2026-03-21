from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

INPUT_DATA_PATH = Path("data/processed/GlobalWeatherRepository_features.parquet")
OUTPUT_RESULTS_PATH = Path("reports/ml_results.csv")
OUTPUT_IMPORTANCE_PATH = Path("reports/ml_feature_importance.csv")
OUTPUT_PREDICTIONS_PATH = Path("reports/ml_predictions.parquet")

RANDOM_STATE = 42

TARGET_COLUMN = "target_temperature_next"
ID_COLUMNS = ["location_name", "country", "last_updated"]

TEMPORAL_CORE_FEATURES = [
    "temperature_celsius",
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "temp_roll_mean_3",
    "temp_roll_std_3",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "day_of_year",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

METEOROLOGICAL_FEATURES = [
    "feels_like_celsius",
    "humidity",
    "cloud",
    "pressure_mb",
    "precip_mm",
    "wind_kph",
    "gust_kph",
    "visibility_km",
    "uv_index",
]

GEOGRAPHIC_FEATURES = [
    "latitude",
    "longitude",
]

AIR_QUALITY_FEATURES = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_us-epa-index",
    "air_quality_gb-defra-index",
]

FEATURE_SETS = {
    "A_temporal_core": TEMPORAL_CORE_FEATURES,
    "B_temporal_meteorology": TEMPORAL_CORE_FEATURES + METEOROLOGICAL_FEATURES,
    "C_temporal_meteorology_geography": TEMPORAL_CORE_FEATURES + METEOROLOGICAL_FEATURES + GEOGRAPHIC_FEATURES,
    "D_full": TEMPORAL_CORE_FEATURES + METEOROLOGICAL_FEATURES + GEOGRAPHIC_FEATURES + AIR_QUALITY_FEATURES,
}


def load_feature_data(path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """Load the engineered dataset used for machine learning experiments."""
    return pd.read_parquet(path)



def prepare_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the modeling dataset and keep only rows with a valid target."""
    df = df.copy()
    df = df.dropna(subset=[TARGET_COLUMN]).sort_values(["location_name", "last_updated"]).reset_index(drop=True)
    return df



def temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a global chronological train/test split."""
    df = df.sort_values("last_updated").reset_index(drop=True)
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df



def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for model evaluation."""
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



def build_pipeline() -> Pipeline:
    """Build the untuned Random Forest pipeline used for feature-set comparison."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )



def train_and_evaluate_feature_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    feature_columns: list[str],
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Train and evaluate the untuned Random Forest model for one feature block."""
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred)
    result_row: dict[str, float | str] = {
        "feature_set": feature_set_name,
        "n_features": len(feature_columns),
        "model": "RandomForestRegressor",
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mape": metrics["mape"],
    }

    predictions_df = test_df[ID_COLUMNS + [TARGET_COLUMN]].copy()
    predictions_df["feature_set"] = feature_set_name
    predictions_df["prediction"] = y_pred

    return result_row, predictions_df



def run_feature_set_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the model across all predefined feature sets."""
    results: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        result_row, predictions_df = train_and_evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            feature_set_name=feature_set_name,
            feature_columns=feature_columns,
        )
        results.append(result_row)
        prediction_frames.append(predictions_df)

    results_df = pd.DataFrame(results).sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return results_df, predictions_df



def tune_best_feature_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_feature_set_name: str,
) -> tuple[pd.DataFrame, Pipeline]:
    """Tune the best feature block with RandomizedSearchCV and evaluate it on the holdout set."""
    feature_columns = FEATURE_SETS[best_feature_set_name]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    time_series_cv = TimeSeriesSplit(n_splits=3)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [12, 18, 24, None],
        "model__min_samples_split": [2, 5, 8, 12],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__max_features": ["sqrt", "log2", 0.8],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    tuned_results_df = pd.DataFrame(
        [
            {
                "feature_set": best_feature_set_name,
                "n_features": len(feature_columns),
                "model": "RandomForestRegressor_tuned",
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
                "best_params": str(best_params),
            }
        ]
    )

    return tuned_results_df, best_model



def compute_permutation_importance_table(
    fitted_pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_set_name: str,
) -> pd.DataFrame:
    """Compute permutation importance for the best tuned model."""
    feature_columns = FEATURE_SETS[feature_set_name]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    importance = permutation_importance(
        estimator=fitted_pipeline,
        X=X_test,
        y=y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
            "feature_set": feature_set_name,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df



def save_outputs(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    """Save machine learning outputs for reporting."""
    OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(OUTPUT_RESULTS_PATH, index=False)
    predictions_df.to_parquet(OUTPUT_PREDICTIONS_PATH, index=False)
    importance_df.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)



if __name__ == "__main__":
    feature_df = load_feature_data()
    modeling_df = prepare_modeling_dataset(feature_df)
    train_df, test_df = temporal_train_test_split(modeling_df, test_size=0.2)

    comparison_results_df, comparison_predictions_df = run_feature_set_comparison(
        train_df=train_df,
        test_df=test_df,
    )

    best_feature_set_name = comparison_results_df.iloc[0]["feature_set"]
    tuned_results_df, best_model = tune_best_feature_set(
        train_df=train_df,
        test_df=test_df,
        best_feature_set_name=best_feature_set_name,
    )

    importance_df = compute_permutation_importance_table(
        fitted_pipeline=best_model,
        test_df=test_df,
        feature_set_name=best_feature_set_name,
    )

    all_results_df = pd.concat([comparison_results_df, tuned_results_df], ignore_index=True)
    all_results_df = all_results_df.sort_values(["mae", "rmse"]).reset_index(drop=True)

    save_outputs(
        results_df=all_results_df,
        predictions_df=comparison_predictions_df,
        importance_df=importance_df,
    )

    print("Machine learning training completed.")
    print(f"Modeling dataset shape: {modeling_df.shape}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("\nFeature set comparison results:")
    print(comparison_results_df)
    print(f"\nBest feature set before tuning: {best_feature_set_name}")
    print("Tuning strategy: RandomizedSearchCV with TimeSeriesSplit(n_splits=3)")
    print("\nTuned model result:")
    print(tuned_results_df)
    print(f"\nResults saved to: {OUTPUT_RESULTS_PATH}")
    print(f"Predictions saved to: {OUTPUT_PREDICTIONS_PATH}")
    print(f"Permutation importance saved to: {OUTPUT_IMPORTANCE_PATH}")