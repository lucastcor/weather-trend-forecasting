from pathlib import Path
import pandas as pd

from src.data.load_data import load_weather_data

PROCESSED_DATA_PATH = Path("data/processed/GlobalWeatherRepository_processed.parquet")

def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the weather dataset for analysis and forecasting.
    
    """
    df= df.copy()

    #remove rows with invalid timestamps just in case
    df = df.dropna(subset=["last_updated"])
    #sort temporally within each location
    df = df.sort_values(["location_name", "last_updated"]).reset_index(drop=True)
    #keep only the columns that are useful for EDA and forecasting
    selected_columns = [
        "country",
        "location_name",
        "latitude",
        "longitude",
        "timezone",
        "last_updated",
        "temperature_celsius",
        "condition_text",
        "wind_kph",
        "wind_degree",
        "pressure_mb",
        "precip_mm",
        "humidity",
        "cloud",
        "feels_like_celsius",
        "visibility_km",
        "uv_index",
        "gust_kph",
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5",
        "air_quality_PM10",
        "air_quality_us-epa-index",
        "air_quality_gb-defra-index",
        "sunrise",
        "sunset",
        "moonrise",
        "moonset",
        "moon_phase",
        "moon_illumination",
    ]
    df = df[selected_columns]
    return df

def save_processed_data(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> None:
    """
    Save the processed weather data to a parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    raw_df = load_weather_data()
    clean_df = preprocess_weather_data(raw_df)
    save_processed_data(clean_df)

    print("Processed dataset saved successfully.")
    print(f"Shape of the processed dataset: {clean_df.shape}")
    print(f"Output path: {PROCESSED_DATA_PATH}")