# 🌫️ AQICast

A Delhi air quality prediction web app built with **Python, Pandas, NumPy and Scikit-learn**, with a **Streamlit** interface.

Live demo: *add your .streamlit.app link here, or delete this line*

## What it does

- Predicts **tomorrow's AQI** for Delhi and shows the CPCB category (Good to Severe)
- Shows **health advice** for the predicted category
- Explains **why**: which features the model relies on most
- **What-if** tool: change today's PM2.5, PM10, wind, humidity and temperature and see how the prediction changes
- **Model performance** tab: compares the models against a simple "tomorrow = today" baseline on days the models never saw
- **Insights** tab: average AQI per month, share of days in each category, weather vs AQI, stubble-burning season vs the rest of the year, and the ten worst days

## How it works

1. **Load and clean (Pandas):** read hourly air quality and weather CSVs, remove duplicate or empty rows, then turn the hourly data into one row per day (averages for pollutants and weather, total for rain). Days with fewer than 16 hours of readings are dropped.
2. **Calculate AQI (NumPy):** the daily PM2.5 and PM10 averages are converted to Indian CPCB AQI sub-indices with `np.interp` using the official breakpoints. The daily AQI is the higher of the two.
3. **Build features (Pandas):** for each day, add yesterday's AQI and the AQI from 2, 3 and 7 days ago, the 3-day and 7-day averages, the change since yesterday, the month, and a flag for stubble-burning season (15 Oct to 30 Nov). The target is **tomorrow's AQI**.
4. **Split by time (Scikit-learn):** the oldest 80% of days are used for training and the newest 20% for testing. The data is never shuffled, because a real forecast can only use the past.
5. **Train models (Scikit-learn):** Linear Regression (with `StandardScaler`) and a Random Forest. Both are compared with a baseline that simply guesses tomorrow will be the same as today.
6. **Evaluate:** MAE, RMSE, R2 and category accuracy (how often the predicted category is exactly right). The best model by MAE is refitted on all days and used for the forecast.

## Tech used

| Library      | Used for                                                              |
| ------------ | --------------------------------------------------------------------- |
| Pandas       | Loading CSVs, hourly to daily conversion, lag features, grouping      |
| NumPy        | AQI breakpoints (`np.interp`), categories, clipping, error maths      |
| Scikit-learn | `LinearRegression`, `RandomForestRegressor`, `StandardScaler`, metrics |
| Streamlit    | The interactive web interface                                         |

## Project structure

```
AQICast/
├── app.py              # Streamlit interface
├── download_data.py    # Downloads Delhi air quality + weather data
├── requirements.txt
├── src/
│   └── predictor.py    # Data loading, AQI, features, models, predictions
└── data/               # air_quality.csv, weather.csv
```

## Run it locally

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

`download_data.py` needs an internet connection and takes a few minutes. To try the app without internet, run `python download_data.py --sample` instead. That creates **made-up** data and the app shows a warning banner while it is in use.

## Dataset

Hourly air quality (PM2.5, PM10, NO2, SO2, CO, ozone) and weather (temperature, humidity, wind, pressure, rain) for New Delhi from August 2022 to yesterday, downloaded from [Open-Meteo](https://open-meteo.com/) (free, no API key). Air quality comes from the CAMS global atmospheric model by the Copernicus Atmosphere Monitoring Service. Open-Meteo data is licensed CC BY 4.0.

## Limitations

- The pollution values come from a **computer model (CAMS)**, not from CPCB ground monitoring stations, so they can differ from the AQI shown on official apps.
- AQI here uses only PM2.5 and PM10. The official CPCB AQI can also use NO2, SO2, CO, O3 and others.
- It predicts only **one day ahead** and uses today's weather, not tomorrow's weather forecast.
- It covers one point in Delhi, and it cannot know about one-off events such as Diwali firecrackers.
- Delhi's AQI changes slowly from day to day, so a "tomorrow = today" guess is already hard to beat. The model's value is in the days when the AQI changes sharply.

## Built with AI assistance

I used an AI assistant (Claude) to help build and learn this project, including the starter structure and parts of the code. I went through the code to understand each function, and I can explain the algorithms used.

## Author

*Your name, year and branch, college*
