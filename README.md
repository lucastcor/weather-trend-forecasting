

# Weather Trend Forecasting Across Global Capital Cities

## Project overview

This project builds a short-term temperature forecasting pipeline using daily observational weather data collected from capital cities around the world. The main objective is to evaluate which modeling approach best captures short-term temperature dynamics and to measure how much predictive value comes from temporal, meteorological, geographic, and environmental features.

The work was structured as an end-to-end data science case study, including data audit, preprocessing, feature engineering, baseline evaluation, supervised machine learning, hyperparameter tuning, permutation importance, and post-model error analysis.

## Problem statement

The central question of the project is:

**Which forecasting approach best models short-term temperature dynamics across global capital cities from daily weather observations?**

A secondary question is also explored:

**Which temporal, geographic, and environmental variables contribute most to predictive performance?**

## Dataset

The dataset used in this project is the **Global Weather Repository**, a daily updating weather dataset with observations from cities around the world. It includes temperature, humidity, precipitation, wind, pressure, cloud cover, visibility, UV index, air-quality measurements, geographic coordinates, and astronomical variables.

Main raw file used in the project:

- `data/raw/GlobalWeatherRepository.csv`

Key observations from the initial audit:

- 130,588 rows and 41 columns in the raw dataset
- no missing values in the original raw file
- no duplicated full rows in the original raw file
- temporal coverage from May 2024 to March 2026
- strong temperature-related signal for short-term forecasting

## Project structure

```text
weather_trend_forecasting/
├── data/
│   ├── auxiliary/
│   ├── processed/
│   └── raw/
├── figures/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_modeling_results.ipynb
├── reports/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── main.py
├── README.md
└── requirements.txt
```

## Methodology

### 1. Data audit

The project started with an audit notebook to validate data quality, inspect the schema, verify temporal coverage, identify candidate target variables, and assess whether the dataset was suitable for forecasting.

The main target chosen for the forecasting task was:

- `temperature_celsius`

The main temporal field used in the project was:

- `last_updated`

### 2. Preprocessing

The preprocessing stage included:

- datetime conversion for `last_updated`
- selection of modeling-relevant variables
- temporal ordering by location
- export of a clean parquet dataset for downstream analysis

### 3. Feature engineering

A dedicated feature engineering stage created:

- calendar features: year, month, day, hour, day of week, day of year
- cyclical encodings: month, hour, and day-of-year sine/cosine transforms
- autoregressive temperature features: lag 1, lag 2, lag 3
- rolling statistics: rolling mean and rolling standard deviation
- next-step target variable: `target_temperature_next`

### 4. Baseline modeling

Before training supervised models, the project established baseline predictors to define the minimum performance level that a machine learning model would need to beat.

Baseline strategies:

- last observation
- rolling mean of the recent window
- naive lag-based forecasts

### 5. Supervised machine learning

The main supervised model used in the project was:

- `RandomForestRegressor`

The model comparison was designed around structured feature blocks instead of simply training on all columns at once.

Feature sets tested:

- **A — Temporal core**
- **B — Temporal + meteorology**
- **C — Temporal + meteorology + geography**
- **D — Full feature set**

After identifying the best feature block, hyperparameter tuning was performed using `RandomizedSearchCV`.

### 6. Model interpretation

To better understand model behavior, the project included:

- correlation screening with the target
- permutation importance for the best tuned model
- observed vs predicted analysis
- residual distribution analysis
- city-level error analysis
- difficult-case and extreme-error inspection
- latitude-band error analysis

## Main results

### Best baseline

The strongest baseline was the persistence-based **last observation** strategy:

- **MAE:** 1.8513
- **RMSE:** 2.7259
- **R²:** 0.9403

This showed that short-term temperature forecasting in this dataset is already strongly driven by recent thermal continuity.

### Best machine learning configuration

The best-performing supervised configuration was:

- **Feature block:** Temporal + meteorology + geography
- **Model:** Tuned Random Forest Regressor

Final tuned performance:

- **MAE:** 1.7439
- **RMSE:** 2.5288
- **R²:** 0.9486

Compared with the best baseline, the tuned Random Forest achieved a measurable improvement, especially in MAE and RMSE.

### What mattered most

Permutation importance showed that the dominant predictors were primarily temperature-history features and closely related weather context:

- `temperature_celsius`
- `feels_like_celsius`
- `temp_roll_mean_3`
- `temp_lag_1`, `temp_lag_2`, `temp_lag_3`
- `latitude`
- `pressure_mb`
- `humidity`
- `uv_index`

### Main analytical finding

The project indicates that short-term temperature forecasting in this dataset is driven primarily by recent thermal history, while meteorological and geographic variables provide incremental predictive value. The full feature set, including air-quality variables, did not outperform the best reduced configuration strongly enough to justify its additional complexity.

## Key conclusions

- A simple persistence-based baseline is already strong for this task.
- Random Forest improved upon the baseline, but the gain came mainly from better feature structure rather than from excessive model complexity.
- Temporal variables remain the primary signal source.
- Meteorological and geographic variables improve forecasting performance modestly but consistently.
- Air-quality variables were useful to explore, but they did not produce the best final configuration for this target.
- Error analysis suggests that abrupt temperature changes are harder to predict than stable short-term trajectories.

## Limitations

This project focuses on **short-term next-step forecasting** using daily observational weather data. It is not a full operational weather forecasting system and it does not attempt to reproduce large-scale physics-based or foundation-model weather systems.

Important limitations include:

- the target is next-step temperature, not a full multi-horizon operational forecast
- the dataset is observational and city-based, not gridded atmospheric reanalysis data
- the modeling approach is data-driven rather than fully physics-informed
- results may vary across locations and climate regimes
- MAPE is reported for completeness, but MAE and RMSE are more reliable for this temperature task because percentage-based error can become unstable near zero values

## How to run the project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd weather_trend_forecasting
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run preprocessing

```bash
python -m src.data.preprocess
```

### 5. Run feature engineering

```bash
python -m src.features.build_features
```

### 6. Run baseline evaluation

```bash
python -m src.models.train_baseline
```

### 7. Run machine learning training

```bash
python -m src.models.train_ml
```

### 8. Open the notebooks

Suggested execution order:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_modeling_results.ipynb`

## Main outputs generated

The project generates the following main artifacts:

- `data/processed/GlobalWeatherRepository_processed.parquet`
- `data/processed/weather_features.parquet`
- `reports/baseline_metrics.csv`
- `reports/baseline_predictions.parquet`
- `reports/ml_results.csv`
- `reports/ml_predictions.parquet`
- `reports/ml_feature_importance.csv`

## Tools and libraries used

Main Python libraries used in the project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pyarrow
- jupyter

## Next steps

Possible extensions for future work include:

- training additional supervised models such as gradient boosting or XGBoost
- building a lightweight ensemble of top-performing models
- extending the target to multi-step or next-day forecasting
- exploring city-specific or region-specific models
- testing uncertainty-aware forecasting approaches
- adding interactive reporting or dashboard layers

## Assessment alignment

This project was intentionally structured to go beyond a minimal forecasting notebook. It includes:

- reproducible preprocessing
- engineered temporal features
- strong baseline comparison
- multiple feature-block experiments
- tuned supervised modeling
- permutation importance
- residual and geographic error analysis
- organized notebooks and reproducible outputs

This makes the repository suitable not only as an assessment submission, but also as a portfolio-quality data science case study.