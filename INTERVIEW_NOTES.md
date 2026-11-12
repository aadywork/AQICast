# AQICast — Interview Notes

## 30-second explanation

AQICast is a prototype that predicts tomorrow's air quality in Delhi. Pandas loads hourly pollution and weather data and turns it into one row per day. NumPy converts PM2.5 and PM10 into the Indian AQI. Scikit-learn trains a Linear Regression and a Random Forest on features like yesterday's AQI, recent averages, wind and season, and tests them on the newest days. I compare them against a simple "tomorrow = today" baseline to check the model is actually useful. Streamlit provides the interface.

## Key concepts

- **NumPy:** A Python library for efficient numerical arrays and calculations.
- **Pandas:** A library for tabular data. A DataFrame is like a programmable spreadsheet.
- **AQI (Air Quality Index):** One number from 0 to 500 that summarises how polluted the air is. In India it uses the CPCB scale: Good, Satisfactory, Moderate, Poor, Very Poor, Severe.
- **`np.interp`:** Linear interpolation. Given the official breakpoints (for example PM2.5 of 30 is AQI 50 and PM2.5 of 60 is AQI 100), it finds the AQI for any value in between.
- **Resampling:** Changing the time step of data. Here, hourly rows become daily rows using the mean (or the sum for rain).
- **Lag feature:** A column holding an earlier value, for example yesterday's AQI. Air pollution carries over from one day to the next, so this is the strongest clue.
- **Rolling average:** The average of the last few days, which smooths out noise.
- **Regression:** Predicting a number (tomorrow's AQI), as opposed to classification, which predicts a label.
- **Linear Regression:** Fits a straight-line relationship between the features and the target. Simple and easy to explain.
- **Random Forest:** Many decision trees that each vote, then the votes are averaged. It can capture non-linear patterns, such as wind only helping when it is strong.
- **StandardScaler:** Puts all features on the same scale so that large numbers (pressure around 1000) do not drown out small ones (rain).
- **Time-based split:** Train on older days and test on newer days. Shuffling would let the model peek at the future, which makes results look better than they really are (data leakage).
- **Baseline:** The simplest possible guess (tomorrow = today). A model is only worth using if it beats this.
- **MAE / RMSE / R2:** MAE is the average mistake in AQI points. RMSE punishes large mistakes more. R2 says how much of the variation the model explains.
- **Feature importance:** For a Random Forest, how much each feature contributed to its decisions.

## Important functions

- `load_data()`: reads the CSVs, removes duplicates and empty rows, converts hourly data to daily data, fills short gaps and calculates the AQI.
- `pm25_to_aqi()` / `pm10_to_aqi()`: convert pollutant concentration to an AQI sub-index with `np.interp`.
- `aqi_category()`: turns an AQI value into a category name using `np.searchsorted`.
- `make_features()`: adds the lag, rolling average, season columns and the target (tomorrow's AQI).
- `build_predictor()`: splits by time, trains the models, scores them against the baseline, picks the best one and refits it on all data.
- `predict_next_day()`: predicts tomorrow's AQI from the newest day of data.
- `what_if()`: changes today's pollution and weather and predicts again.
- `feature_importance()`: shows what the model pays attention to.

## Questions you may be asked

- **Why not shuffle the data for the train/test split?** Because we are forecasting the future. Shuffling would train on days after the test days, so the score would look better than reality.
- **Why compare with a baseline?** AQI changes slowly, so "tomorrow = today" is already a decent guess. If the model cannot beat it, it adds nothing.
- **Why two models?** Linear Regression is a simple, explainable starting point. Random Forest can capture non-linear effects. Comparing them shows which one fits this data better.
- **Why is yesterday's AQI so important?** Pollutants stay in the air for days, especially in winter when the air is still.
- **Why use the maximum of the PM2.5 and PM10 sub-indices?** That is how the CPCB defines the AQI: the worst pollutant decides the number.
- **What is the stubble-burning feature?** A 0/1 flag for 15 Oct to 30 Nov, when crop burning in nearby states adds to Delhi's smog. It lets the model learn that this period behaves differently.
- **Can you trust the data?** It comes from a satellite and model based dataset (CAMS), not ground stations, so values can differ from official CPCB readings.
- **What would you improve?** See below.

## Limitations / future improvements

- Pollution data is modelled (CAMS), not from CPCB ground stations. Using real station data would be more accurate.
- It predicts one day ahead using today's weather. Adding tomorrow's weather forecast would help.
- Could add satellite fire counts for stubble burning, festival dates (Diwali) and traffic data.
- Could try Gradient Boosting, tune the Random Forest with cross-validation for time series (`TimeSeriesSplit`), and predict several days ahead.
- Could show prediction ranges instead of a single number.
