"""Downloads Delhi air quality and weather data from Open-Meteo (free, no API key).

Usage:
    python download_data.py            # real data, Aug 2022 up to yesterday
    python download_data.py --sample   # made-up demo data, works without internet

Creates data/air_quality.csv and data/weather.csv (one row per hour).
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_FLAG = DATA_DIR / "SAMPLE_DATA.txt"

LAT, LON = 28.6139, 77.2090          # New Delhi
START = date(2022, 8, 1)              # air quality history starts around here

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

AIR_VARS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone"]
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "surface_pressure", "precipitation"]


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def fetch(url, variables, start, end):
    """Download hourly data in 90-day pieces (the APIs limit long requests)."""
    import requests

    pieces = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=89), end)
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(variables),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": "Asia/Kolkata",
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        pieces.append(pd.DataFrame(response.json()["hourly"]))
        print(f"  downloaded {chunk_start} to {chunk_end}")
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # be polite to the free API
    return pd.concat(pieces, ignore_index=True)


def download_real():
    end = date.today() - timedelta(days=1)
    DATA_DIR.mkdir(exist_ok=True)

    print("Air quality:")
    fetch(AIR_URL, AIR_VARS, START, end).to_csv(DATA_DIR / "air_quality.csv", index=False)
    print("Weather:")
    fetch(WEATHER_URL, WEATHER_VARS, START, end).to_csv(DATA_DIR / "weather.csv", index=False)

    SAMPLE_FLAG.unlink(missing_ok=True)
    print("Done. Saved data/air_quality.csv and data/weather.csv")


# ---------------------------------------------------------------------------
# Sample data (made up, for testing without internet)
# ---------------------------------------------------------------------------
def download_sample():
    """Create fake but realistic-looking Delhi data: dirty winters, clean monsoon."""
    rng = np.random.default_rng(42)
    end = date.today() - timedelta(days=1)
    days = pd.date_range(START, end, freq="D")
    n = len(days)
    doy = days.dayofyear.to_numpy()
    season = lambda peak_day: 2 * np.pi * (doy - peak_day) / 365

    # Daily weather
    temp = 24 + 10 * np.sin(season(60)) + rng.normal(0, 1.5, n)
    humidity = np.clip(55 + 22 * np.sin(season(125)) + rng.normal(0, 8, n), 15, 98)
    wind = np.clip((10 + 3 * np.sin(season(60))) * np.exp(rng.normal(0, 0.35, n)), 1, None)
    pressure = 1000 - 8 * np.sin(season(60)) + rng.normal(0, 2, n)
    monsoon = (doy >= 182) & (doy <= 273)
    rainy = rng.random(n) < np.where(monsoon, 0.5, 0.05)
    rain = np.where(rainy, rng.exponential(8, n), 0.0)

    # Daily PM2.5: seasonal level + yesterday's value + weather effects
    winter_peak = np.cos(season(330))   # +1 in late November, -1 in late May
    log_pm25 = np.zeros(n)
    latent = 0.0
    for i in range(n):
        latent = 0.8 * latent + rng.normal(0, 0.25)
        stubble = 0.3 if 293 <= doy[i] <= 320 else 0.0
        log_pm25[i] = (np.log(85) + 0.75 * winter_peak[i] + latent + stubble
                       - 0.05 * (wind[i] - 10) - (0.5 if rain[i] > 2 else 0.0))
    pm25 = np.clip(np.exp(log_pm25), 5, 600)
    pm10 = pm25 * 1.9 * np.exp(rng.normal(0, 0.1, n))
    no2 = np.clip(25 + 0.12 * pm25 + rng.normal(0, 5, n), 5, None)
    so2 = np.clip(10 + 0.02 * pm25 + rng.normal(0, 2, n), 1, None)
    co = np.clip(600 + 4 * pm25 + rng.normal(0, 80, n), 100, None)
    o3 = np.clip(40 + 1.5 * (temp - 24) + rng.normal(0, 8, n), 5, None)

    # Daily -> hourly, with a day/night pattern (PM peaks before sunrise)
    hours = np.tile(np.arange(24), n)
    daily = lambda values: np.repeat(values, 24)
    night = 1 + 0.25 * np.cos(2 * np.pi * (hours - 5) / 24)
    sun = np.sin(2 * np.pi * (hours - 9) / 24)
    noise = lambda scale: np.exp(rng.normal(0, scale, n * 24))
    time_index = pd.date_range(START, periods=n * 24, freq="h")

    air = pd.DataFrame({
        "time": time_index,
        "pm10": daily(pm10) * night * noise(0.05),
        "pm2_5": daily(pm25) * night * noise(0.05),
        "carbon_monoxide": daily(co) * night * noise(0.05),
        "nitrogen_dioxide": daily(no2) * night * noise(0.05),
        "sulphur_dioxide": daily(so2) * noise(0.05),
        "ozone": daily(o3) * (1 + 0.4 * sun) * noise(0.05),
    })
    air.iloc[:, 1:] = air.iloc[:, 1:].round(1)   # round everything except the time column

    weather = pd.DataFrame({
        "time": time_index,
        "temperature_2m": daily(temp) + 5 * sun,
        "relative_humidity_2m": np.clip(daily(humidity) - 15 * sun, 5, 100),
        "wind_speed_10m": daily(wind) * (1 + 0.3 * sun),
        "surface_pressure": daily(pressure) + 0.8 * np.cos(2 * np.pi * hours / 12),
        "precipitation": daily(rain) / 24,
    })
    weather.iloc[:, 1:] = weather.iloc[:, 1:].round(2)

    DATA_DIR.mkdir(exist_ok=True)
    air.to_csv(DATA_DIR / "air_quality.csv", index=False)
    weather.to_csv(DATA_DIR / "weather.csv", index=False)
    SAMPLE_FLAG.write_text(
        "The CSV files in this folder are MADE-UP sample data, not real measurements.\n"
        "Run `python download_data.py` with internet to replace them with real data.\n"
    )
    print("Created SAMPLE data (not real). Run without --sample to get real data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Delhi AQI and weather data.")
    parser.add_argument("--sample", action="store_true",
                        help="create made-up demo data instead of downloading")
    args = parser.parse_args()
    if args.sample:
        download_sample()
    else:
        try:
            download_real()
        except Exception as error:
            print(f"Download failed: {error}")
            print("Check your internet connection, or use --sample for demo data.")
