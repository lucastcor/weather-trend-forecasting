from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DATA_PATH = Path("data/processed/GlobalWeatherRepository_processed.parquet")
OUTPUT_DATA_PATH = Path("data/processed/GlobalWeatherRepository_features.parquet")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based temporal features from timestamp column.

    """
    df = df.copy()
    df["year"]= df["last_updated"].dt.year
    df["month"]= df["last_updated"].dt.month
    df["day"]= df["last_updated"].dt.day
    df["hour"]= df["last_updated"].dt.hour
    df["day_of_week"]= df["last_updated"].dt.dayofweek
    df["day_of_year"]= df["last_updated"].dt.dayofyear

    return df

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclical temporal variables using sine and cosine transformations.
    """
    df = df.copy()

    df["month_sin"] = np.sin(2 * np.pi * df["month"]/ 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]/ 12)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"]/ 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"]/ 24)

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"]/ 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"]/ 365)

    return df

def add_temperature_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag and rolling feature for temperature by location.
    """
    df= df.copy()

    grouped = df.groupby("location_name")["temperature_celsius"]
    df["temp_lag_1"] = grouped.shift(1)
    df["temp_lag_2"] = grouped.shift(2)
    df["temp_lag_3"] = grouped.shift(3)
    df["temp_roll_mean_3"]= grouped.transform(lambda s: s.shift(1).rolling(window=3).mean())
    df["temp_roll_std_3"]= grouped.transform(lambda s: s.shift(1).rolling(window=3).std())

    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the next-step temperature target for each location.
    """
    df = df.copy()

    df["target_temperature_next"] = df.groupby("location_name")["temperature_celsius"].shift(-1)

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df= df.copy()
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_temperature_lag_features(df)
    df = add_target(df)
    return df

def save_features_data(df: pd.DataFrame, output_path: Path= OUTPUT_DATA_PATH) -> None:
    """
    Save engineered features to a parquet file for modeling.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    df = pd.read_parquet(INPUT_DATA_PATH)
    features_df = build_features(df)
    save_features_data(features_df)
    print("Feature dataset saved successfully.")
    print(f"Shape of the feature dataset: {features_df.shape}")
    print(f"Output path: {OUTPUT_DATA_PATH}")
