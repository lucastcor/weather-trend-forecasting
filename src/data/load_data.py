from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/GlobalWeatherRepository.csv")

def load_weather_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """"
    Load the raw weather datase and apply basic type parsing.
    """
    df = pd.read_csv(path)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df
