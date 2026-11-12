"""AQICast: data loading, feature building and models for predicting
tomorrow's Delhi AQI with Pandas, NumPy and Scikit-learn."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Indian AQI (CPCB) breakpoints: pollutant concentration -> AQI value
# ---------------------------------------------------------------------------
PM25_POINTS = [0, 30, 60, 90, 120, 250, 380]
PM10_POINTS = [0, 50, 100, 250, 350, 430, 510]
AQI_POINTS = [0, 50, 100, 200, 300, 400, 500]

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CATEGORY_EDGES = [50, 100, 200, 300, 400]  # upper limit of each category
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ADVICE = {
    "Good": "Air is clean. Enjoy outdoor activities.",
    "Satisfactory": "Fine for most people. Very sensitive people may feel mild discomfort.",
    "Moderate": "People with asthma or heart disease should limit long outdoor exertion.",
    "Poor": "Avoid prolonged outdoor activity. Children and older adults should stay indoors.",
    "Very Poor": "Wear an N95 mask outside and avoid outdoor exercise. Use an air purifier indoors.",
    "Severe": "Health emergency conditions. Stay indoors and avoid all outdoor activity.",
}

# ---------------------------------------------------------------------------
# Features used by the models
# ---------------------------------------------------------------------------
FEATURES = [
    "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3",       # today's air
    "temp", "humidity", "wind", "pressure", "rain",          # today's weather
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7",          # past AQI values
    "aqi_avg3", "aqi_avg7", "aqi_change",                    # recent trend
    "month", "stubble_season",                               # time of year
]


# ---------------------------------------------------------------------------
# AQI calculation
# ---------------------------------------------------------------------------
def pm25_to_aqi(pm25):
    """Convert PM2.5 (ug/m3, daily average) to an AQI sub-index."""
    return np.interp(pm25, PM25_POINTS, AQI_POINTS)


def pm10_to_aqi(pm10):
    """Convert PM10 (ug/m3, daily average) to an AQI sub-index."""
    return np.interp(pm10, PM10_POINTS, AQI_POINTS)


def aqi_category(aqi):
    """Return the CPCB category name for one AQI value or an array of values."""
    index = np.searchsorted(CATEGORY_EDGES, aqi, side="left")
    names = np.array(CATEGORIES)[index]
    return str(names) if np.ndim(aqi) == 0 else names


def is_sample_data(data_dir=DATA_DIR):
    """True if the data folder holds synthetic sample data, not real data."""
    return (Path(data_dir) / "SAMPLE_DATA.txt").exists()


# ---------------------------------------------------------------------------
# Load and clean
# ---------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    """Read the hourly CSVs, turn them into one row per day and add the AQI."""
    data_dir = Path(data_dir)
    air = pd.read_csv(data_dir / "air_quality.csv", parse_dates=["time"])
    weather = pd.read_csv(data_dir / "weather.csv", parse_dates=["time"])

    # Remove duplicate hours and hours with no PM readings
    air = air.drop_duplicates(subset="time").dropna(subset=["pm2_5", "pm10"])
    weather = weather.drop_duplicates(subset="time").dropna()

    # Hourly -> daily (averages for air, average + total rain for weather)
    air = air.set_index("time")
    hours_per_day = air["pm2_5"].resample("D").count()
    air_daily = air.resample("D").mean()
    air_daily = air_daily[hours_per_day >= 16]  # CPCB needs at least 16 hours

    weather = weather.set_index("time")
    weather_daily = weather.resample("D").agg({
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean",
        "surface_pressure": "mean",
        "precipitation": "sum",
    })

    df = air_daily.join(weather_daily, how="inner")
    df = df.rename(columns={
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co",
        "ozone": "o3",
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind",
        "surface_pressure": "pressure",
        "precipitation": "rain",
    })

    # Make sure every day is present, fill gaps of up to 3 days, drop the rest
    df = df.asfreq("D").interpolate(limit=3).dropna()
    df.index.name = "date"
    df = df.reset_index()

    # Daily AQI = the worse of the PM2.5 and PM10 sub-indices
    df["aqi"] = np.maximum(pm25_to_aqi(df["pm2_5"]), pm10_to_aqi(df["pm10"]))
    df["category"] = aqi_category(df["aqi"].to_numpy())
    return df


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def make_features(df):
    """Add lag, trend and season columns, and the target (tomorrow's AQI)."""
    f = df.sort_values("date").reset_index(drop=True).copy()

    for days in (1, 2, 3, 7):
        f[f"aqi_lag{days}"] = f["aqi"].shift(days)
    f["aqi_avg3"] = f["aqi"].rolling(3).mean()
    f["aqi_avg7"] = f["aqi"].rolling(7).mean()
    f["aqi_change"] = f["aqi"] - f["aqi_lag1"]

    f["month"] = f["date"].dt.month
    month_day = f["date"].dt.month * 100 + f["date"].dt.day
    f["stubble_season"] = ((month_day >= 1015) & (month_day <= 1130)).astype(int)

    f["target"] = f["aqi"].shift(-1)  # tomorrow's AQI
    return f


# ---------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------
def _score(name, actual, predicted):
    """Collect the error metrics for one model."""
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Category accuracy %": 100 * np.mean(
            aqi_category(np.asarray(actual)) == aqi_category(np.asarray(predicted))
        ),
    }


def build_predictor(df, test_fraction=0.2):
    """Train the models on the older days, test them on the newest days,
    then refit the best one on all the data."""
    feat = make_features(df)
    data = feat.dropna(subset=FEATURES + ["target"])

    # Time-based split: never test on days that came before the training days
    split = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
    }

    baseline = "Baseline (tomorrow = today)"
    rows = [_score(baseline, y_test, test["aqi"].to_numpy())]
    predictions = pd.DataFrame({"date": test["date"].to_numpy(),
                                "Actual": y_test.to_numpy(),
                                baseline: test["aqi"].to_numpy()})

    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = np.clip(model.predict(X_test), 0, 500)
        rows.append(_score(name, y_test, predicted))
        predictions[name] = predicted

    results = pd.DataFrame(rows).set_index("Model").round(2)
    best_name = results.drop(index=baseline)["MAE"].idxmin()

    best_model = models[best_name]
    best_model.fit(data[FEATURES], data["target"])  # refit on all history

    return {
        "features": feat,
        "results": results,
        "predictions": predictions,
        "best_name": best_name,
        "model": best_model,
        "baseline_name": baseline,
        "latest": feat.iloc[[-1]],  # newest day, used to predict tomorrow
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict_next_day(model, latest):
    """Predict tomorrow's AQI from the newest day of data."""
    value = model.predict(latest[FEATURES])[0]
    return float(np.clip(value, 0, 500))


def what_if(model, latest, pm2_5, pm10, wind, humidity, temp):
    """Predict tomorrow's AQI if today's pollution and weather were different."""
    row = latest.copy()
    row["pm2_5"] = pm2_5
    row["pm10"] = pm10
    row["wind"] = wind
    row["humidity"] = humidity
    row["temp"] = temp
    row["aqi"] = max(pm25_to_aqi(pm2_5), pm10_to_aqi(pm10))
    row["aqi_change"] = row["aqi"] - row["aqi_lag1"]
    return predict_next_day(model, row)


def feature_importance(model):
    """Share of the prediction explained by each feature (adds up to 1)."""
    if hasattr(model, "feature_importances_"):          # Random Forest
        scores = model.feature_importances_
    else:                                               # Linear Regression pipeline
        scores = np.abs(model[-1].coef_)                # features are scaled, so comparable
    return pd.Series(scores / scores.sum(), index=FEATURES).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------
def monthly_average(df):
    """Average AQI for each calendar month (Jan to Dec)."""
    by_month = df.groupby(df["date"].dt.month)["aqi"].mean().reindex(range(1, 13))
    by_month.index = MONTH_NAMES
    return by_month.round(1)


def category_share(df):
    """Percentage of days in each AQI category."""
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    return (100 * counts / counts.sum()).round(1)


def weather_correlation(df):
    """How strongly each weather column moves with AQI (-1 to +1)."""
    cols = ["aqi", "temp", "humidity", "wind", "pressure", "rain"]
    return df[cols].corr()["aqi"].drop("aqi").round(2)


def stubble_comparison(features):
    """Average AQI during stubble-burning season (15 Oct to 30 Nov) vs the rest of the year."""
    groups = features.groupby("stubble_season")["aqi"].mean()
    return {
        "Stubble season": round(groups.get(1, float("nan")), 1),
        "Rest of year": round(groups.get(0, float("nan")), 1),
    }


def worst_days(df, n=10):
    """The n days with the highest AQI."""
    cols = ["date", "aqi", "category", "pm2_5", "pm10", "wind"]
    top = df.nlargest(n, "aqi")[cols].copy()
    top["date"] = top["date"].dt.date
    return top.round(1).reset_index(drop=True)
