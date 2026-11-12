import altair as alt
import pandas as pd
import streamlit as st

from src.predictor import (
    ADVICE,
    CATEGORIES,
    CATEGORY_EDGES,
    MONTH_NAMES,
    aqi_category,
    build_predictor,
    category_share,
    feature_importance,
    is_sample_data,
    load_data,
    monthly_average,
    predict_next_day,
    stubble_comparison,
    weather_correlation,
    what_if,
    worst_days,
)

st.set_page_config(page_title="AQICast", page_icon="🌫️", layout="wide")

CATEGORY_COLORS = {
    "Good": "#1B8A5A",
    "Satisfactory": "#6E9B1F",
    "Moderate": "#B58900",
    "Poor": "#D9730D",
    "Very Poor": "#C0392B",
    "Severe": "#7E0023",
}
COLOR_LIST = [CATEGORY_COLORS[c] for c in CATEGORIES]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .hero {border-radius: 16px; padding: 26px 30px; color: white;}
    .hero .label {opacity: .85; font-size: .95rem; letter-spacing: .04em;}
    .hero .big {font-size: 4.2rem; font-weight: 800; line-height: 1.05;}
    .hero .cat {font-size: 1.5rem; font-weight: 700;}
    .hero .advice {margin-top: 8px; font-size: 1.05rem;}
    .scale {display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin-top: 18px;}
    .scale div {flex: 1;}
    .pointer {position: relative; height: 16px;}
    .pointer span {position: absolute; transform: translateX(-50%); font-size: 1.1rem;}
    div[data-testid="stMetric"] {background: #1A1F2B; padding: 14px 16px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_predictor():
    return build_predictor(get_data())


def category_scale():
    """Return the domain/range pair so altair colours bars by AQI category."""
    return alt.Scale(domain=CATEGORIES, range=COLOR_LIST)


def hero_card(predicted, category, tomorrow_date, today_aqi):
    """Big coloured card with the forecast, advice and a position marker on the AQI scale."""
    colour = CATEGORY_COLORS[category]
    arrow = "▲" if predicted > today_aqi else "▼"
    segments = "".join(f"<div style='background:{c}'></div>" for c in COLOR_LIST)
    position = min(predicted, 500) / 500 * 100
    html = (
        f"<div class='hero' style='background:linear-gradient(135deg,{colour},#111)'>"
        f"<div class='label'>PREDICTED AQI FOR {tomorrow_date:%A, %d %b}</div>"
        f"<div class='big'>{predicted:.0f} <span style='font-size:1.6rem'>"
        f"{arrow} {abs(predicted - today_aqi):.0f}</span></div>"
        f"<div class='cat'>{category}</div>"
        f"<div class='advice'>{ADVICE[category]}</div>"
        f"<div class='scale'>{segments}</div>"
        f"<div class='pointer'><span style='left:{position}%'>▲</span></div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def band_chart(frame, y_max):
    """Line chart of AQI with the six category colours as faint background bands."""
    edges = [0] + CATEGORY_EDGES + [max(500, y_max)]
    bands = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "category": CATEGORIES})
    background = alt.Chart(bands).mark_rect(opacity=0.18).encode(
        y="start:Q", y2="end:Q", color=alt.Color("category:N", scale=category_scale(), legend=None))
    line = alt.Chart(frame).mark_line(point=True, color="#FFFFFF").encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("aqi:Q", title="AQI", scale=alt.Scale(domain=[0, y_max])),
        tooltip=["date:T", alt.Tooltip("aqi:Q", format=".0f")])
    return (background + line).properties(height=300)


st.title("🌫️ AQICast")
st.caption("Tomorrow's Delhi air quality, predicted with NumPy, Pandas and Scikit-learn")

try:
    df = get_data()
except FileNotFoundError:
    st.error("Data files not found. Run `python download_data.py` first "
             "(or `python download_data.py --sample` for demo data).")
    st.stop()

if is_sample_data():
    st.warning("You are looking at SAMPLE data (made up, not real measurements). "
               "Run `python download_data.py` to load real data.")

predictor = get_predictor()
model = predictor["model"]
latest = predictor["latest"]
results = predictor["results"]
best_name = predictor["best_name"]
baseline_name = predictor["baseline_name"]

with st.sidebar:
    st.header("About AQICast")
    st.write("Predicts **tomorrow's AQI** from today's pollution, weather and "
             "the last week of AQI values.")
    st.metric("Days of data", len(df))
    st.write(f"**From:** {df['date'].min():%d %b %Y}")
    st.write(f"**To:** {df['date'].max():%d %b %Y}")
    st.write(f"**Model in use:** {best_name}")
    st.caption("AQI uses the Indian CPCB scale, calculated from daily PM2.5 and PM10.")

tab_predict, tab_model, tab_insights = st.tabs(
    ["🔮 Tomorrow's forecast", "📊 Model performance", "📈 Insights"]
)

# ---------------------------------------------------------------- Forecast
with tab_predict:
    today = latest.iloc[0]
    tomorrow_date = today["date"] + pd.Timedelta(days=1)
    predicted = predict_next_day(model, latest)
    predicted_category = aqi_category(predicted)
    typical_error = results.loc[best_name, "MAE"]

    hero_col, side_col = st.columns([3, 2])
    with hero_col:
        hero_card(predicted, predicted_category, tomorrow_date, today["aqi"])
    with side_col:
        st.metric(f"Today ({today['date']:%d %b})", f"{today['aqi']:.0f}", today["category"],
                  delta_color="off")
        st.metric("Typical error of the model", f"± {typical_error:.0f} AQI")
        st.metric("Wind today", f"{today['wind']:.1f} km/h",
                  help="Wind and rain usually clear the air.")

    st.subheader("Last 30 days")
    recent = df.tail(30)[["date", "aqi"]]
    st.altair_chart(band_chart(recent, max(500, float(recent["aqi"].max()))),
                    use_container_width=True)

    left, right = st.columns(2)
    with left:
        with st.expander("Why this prediction?", expanded=True):
            importance = (feature_importance(model).head(8) * 100).round(1)
            imp = importance.rename("Importance %").rename_axis("Feature").reset_index()
            chart = alt.Chart(imp).mark_bar(color="#4FC3F7").encode(
                x=alt.X("Importance %:Q"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", "Importance %"])
            st.altair_chart(chart.properties(height=260), use_container_width=True)
            st.caption("`aqi_lag1` = yesterday's AQI, `aqi_avg3` = 3-day average.")
    with right:
        with st.expander("Try a what-if", expanded=True):
            w1, w2 = st.columns(2)
            pm25 = w1.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(min(round(today["pm2_5"]), 500)))
            pm10 = w1.slider("PM10 (µg/m³)", 0.0, 1000.0, float(min(round(today["pm10"]), 1000)))
            wind = w2.slider("Wind (km/h)", 0.0, 40.0, float(min(round(today["wind"]), 40)))
            humidity = w2.slider("Humidity (%)", 0.0, 100.0,
                                 float(min(round(today["humidity"]), 100)))
            temp = w2.slider("Temperature (°C)", 0.0, 50.0,
                             float(min(max(round(today["temp"]), 0), 50)))
            scenario = what_if(model, latest, pm25, pm10, wind, humidity, temp)
            st.metric("Tomorrow's AQI in this scenario", f"{scenario:.0f}",
                      aqi_category(scenario), delta_color="off")

# ----------------------------------------------------------- Model performance
with tab_model:
    st.write("The newest 20% of days were kept aside as a test set. Each model was trained only "
             "on older days, then asked to predict tomorrow's AQI for the test days.")

    base_mae = results.loc[baseline_name, "MAE"]
    best_mae = results.loc[best_name, "MAE"]
    gain = 100 * (base_mae - best_mae) / base_mae
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", f"{base_mae:.1f}")
    c2.metric(f"{best_name} MAE", f"{best_mae:.1f}", f"{-gain:.1f}% error", delta_color="inverse")
    c3.metric("Category accuracy", f"{results.loc[best_name, 'Category accuracy %']:.0f}%")
    if gain > 0:
        st.success(f"{best_name} has {gain:.1f}% lower average error than the baseline.")
    else:
        st.warning(f"{best_name} does not beat the baseline of guessing tomorrow = today.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Average error (lower is better)")
        mae = results["MAE"].rename("MAE").reset_index()
        bars = alt.Chart(mae).mark_bar().encode(
            x=alt.X("MAE:Q", title="MAE (AQI points)"),
            y=alt.Y("Model:N", sort="x", title=None),
            color=alt.condition(alt.datum.Model == best_name,
                                alt.value("#4FC3F7"), alt.value("#5A6272")),
            tooltip=["Model", "MAE"])
        st.altair_chart(bars.properties(height=200), use_container_width=True)
    with right:
        st.subheader("All metrics")
        st.dataframe(results, use_container_width=True)
    st.caption("MAE = average mistake in AQI points. RMSE punishes big mistakes more. "
               "R2 = how much of the ups and downs the model explains (1 is perfect).")

    predictions = predictor["predictions"]
    days_to_show = st.slider("Days of the test period to show", 10, len(predictions),
                             min(60, len(predictions)))
    shown = predictions.tail(days_to_show)[["date", "Actual", best_name]]
    long = shown.melt("date", var_name="Series", value_name="AQI")
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None), y="AQI:Q",
        color=alt.Color("Series:N", scale=alt.Scale(range=["#FFFFFF", "#4FC3F7"])),
        tooltip=["date:T", "Series", alt.Tooltip("AQI:Q", format=".0f")])
    st.altair_chart(lines.properties(height=300), use_container_width=True)

# ---------------------------------------------------------------- Insights
with tab_insights:
    years = ["All years"] + sorted(df["date"].dt.year.unique().tolist(), reverse=True)
    choice = st.selectbox("Year", years)
    subset = df if choice == "All years" else df[df["date"].dt.year == choice]

    left, right = st.columns(2)
    with left:
        st.subheader("Average AQI by month")
        monthly = monthly_average(subset).dropna().rename("AQI").rename_axis("Month").reset_index()
        monthly["Category"] = aqi_category(monthly["AQI"].to_numpy())
        chart = alt.Chart(monthly).mark_bar().encode(
            x=alt.X("Month:N", sort=MONTH_NAMES, title=None),
            y=alt.Y("AQI:Q"),
            color=alt.Color("Category:N", scale=category_scale(), legend=None),
            tooltip=["Month", "AQI", "Category"])
        st.altair_chart(chart.properties(height=280), use_container_width=True)
    with right:
        st.subheader("Share of days in each category (%)")
        shares = category_share(subset).rename("Share").rename_axis("Category").reset_index()
        chart = alt.Chart(shares).mark_bar().encode(
            x=alt.X("Category:N", sort=CATEGORIES, title=None),
            y=alt.Y("Share:Q", title="% of days"),
            color=alt.Color("Category:N", scale=category_scale(), legend=None),
            tooltip=["Category", "Share"])
        st.altair_chart(chart.properties(height=280), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Weather vs AQI")
        corr = weather_correlation(subset).rename("Correlation").rename_axis("Weather").reset_index()
        chart = alt.Chart(corr).mark_bar().encode(
            x=alt.X("Weather:N", sort=None, title=None),
            y=alt.Y("Correlation:Q", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.Correlation > 0,
                                alt.value("#D9730D"), alt.value("#4FC3F7")),
            tooltip=["Weather", "Correlation"])
        st.altair_chart(chart.properties(height=260), use_container_width=True)
        st.caption("Positive = AQI rises with it, negative = AQI falls with it.")
    with right:
        st.subheader("Stubble-burning season")
        season = stubble_comparison(predictor["features"])
        s1, s2 = st.columns(2)
        s1.metric("Avg AQI, 15 Oct to 30 Nov", f"{season['Stubble season']:.0f}")
        s2.metric("Avg AQI, rest of year", f"{season['Rest of year']:.0f}")
        st.caption("Crop-residue burning in Punjab and Haryana plus winter weather "
                   "traps smoke over Delhi.")

    st.subheader("Ten worst days")
    st.dataframe(worst_days(subset), hide_index=True, use_container_width=True)