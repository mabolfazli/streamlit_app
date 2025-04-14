import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from io import StringIO
import requests


# Set the page layout to wide
st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(StringIO(res.text))
    df['TIME'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
    return df

# get token and make url
path_data = 'https://raw.githubusercontent.com/adamcohen3/caseload_forecasting/refs/heads/main/Processed_Data/individual_courts/'
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
headers = {'Authorization': f'token {token}'}
url = "https://raw.githubusercontent.com/Hawaii-State-Judiciary/caseload_forecasting/refs/heads/main/Processed_Data/individual_courts/monthly_case_filings_1998-2024.csv"
res = requests.get(url, headers=headers)
df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
case_types = df["CASE_TYPE_NAME"].unique()
selected_case_type = st.sidebar.selectbox("Select Case Type", case_types)

courts = df[df["CASE_TYPE_NAME"] == selected_case_type]["COURT"].unique()
selected_court = st.sidebar.selectbox("Select Court", courts)

stationarity = [False, True]
selected_stationary = st.sidebar.selectbox("Stationarity", stationarity)
seasonality = st.sidebar.slider("Seasonal Period", min_value=2, max_value=12, value=12)

# GBR hyperparameter inputs
max_depth = st.sidebar.number_input("GBR: Max Depth", min_value=1, max_value=10, value=3, step=1)
n_estimators = st.sidebar.number_input("GBR: Number of Estimators", min_value=10, max_value=200, value=50, step=10)

# Filter data
df_case = df[(df["CASE_TYPE_NAME"] == selected_case_type) & (df["COURT"] == selected_court)].copy()
df_case = df_case.set_index('TIME')
df_case = df_case.asfreq('MS')
df_case["CASE_COUNT"].fillna(0, inplace=True)

# Create lagged features
df_case['LAGGED_CASE_COUNT_1M'] = df_case['CASE_COUNT'].shift(1)
df_case['LAGGED_CASE_COUNT_2M'] = df_case['CASE_COUNT'].shift(2)
df_case['LAGGED_CASE_COUNT_3M'] = df_case['CASE_COUNT'].shift(3)
df_case['LAGGED_CASE_COUNT_12M'] = df_case['CASE_COUNT'].shift(12)

# Create rolling features
df_case['LAGGED_MEAN_CASE_COUNT_3M'] = df_case['CASE_COUNT'].shift(1).rolling(window=3).mean()
df_case['LAGGED_MEAN_CASE_COUNT_12M'] = df_case['CASE_COUNT'].shift(1).rolling(window=12).mean()
df_case['LAGGED_MAX_CASE_COUNT_12M'] =  df_case['CASE_COUNT'].shift(1).rolling(window=12).max()
df_case['LAGGED_MIN_CASE_COUNT_12M'] =  df_case['CASE_COUNT'].shift(1).rolling(window=12).min()

df_case.dropna(inplace=True)

# Split data into training/testing
X = df_case.iloc[:, 5:]
y = df_case["CASE_COUNT"]
split_idx = int(0.9 * len(y))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train models
gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=123)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
gbr_pred = np.round(gbr_pred, 0).astype(int)

arima = pm.auto_arima(y_train, error_action="ignore", suppress_warnings=True, maxiter=500, m=seasonality, stationary=selected_stationary)
arima_pred = arima.predict(n_periods=len(y_test))
arima_pred = np.round(arima_pred, 0).astype(int)

holt = ExponentialSmoothing(y_train, seasonal_periods=seasonality, trend="add", seasonal="mul", damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
holt_pred = holt.forecast(len(y_test))
holt_pred = np.round(holt_pred, 0).astype(int)

st.subheader(f"Forecasting Using Different Timeseries Models- {selected_case_type} {selected_court}")

# Create an interactive plot with Plotly
fig = go.Figure()

# Add traces for each dataset
fig.add_trace(go.Scatter(
    x=y_train.index, y=y_train,
    mode="lines+markers", name="Train",
    hoverinfo="x+y", line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=y_test.index, y=y_test,
    mode="lines+markers", name="Test",
    hoverinfo="x+y", line=dict(color="gray")
))

fig.add_trace(go.Scatter(
    x=y_test.index, y=gbr_pred,
    mode="lines+markers", name="GBR Prediction",
    hoverinfo="x+y", line=dict(color="orange")
))

fig.add_trace(go.Scatter(
    x=y_test.index, y=arima_pred,
    mode="lines+markers", name="ARIMA Prediction",
    hoverinfo="x+y", line=dict(color="red")
))

fig.add_trace(go.Scatter(
    x=y_test.index, y=holt_pred,
    mode="lines+markers", name="Holt Prediction",
    hoverinfo="x+y", line=dict(color="green")
))

# Customize layout for better readability
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Case Count",
    hovermode="x unified",  # Ensures all tooltips appear together
    template="plotly_white",  # Clean background
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    height=800,
    xaxis=dict(rangeslider=dict(visible=True, thickness=0.05), type="date")
)

# Create a two-column layout with a gap in between
col1, spacer, col2 = st.columns([3, 0.2, 2])

# Display the plot in the first column
with col1:
    st.plotly_chart(fig, use_container_width=True)

# Display the forecast results in the second column
with col2:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True) # add space above table to align it with the figure
    results_df = pd.DataFrame({
        "Time": y_test.index.strftime("%Y-%m-%d"),  # Keep only YYYY-MM-DD
        "Actual": y_test.values,
        "GBR": gbr_pred,
        "ARIMA": arima_pred,
        "Holt": holt_pred,
    }).reset_index(drop=True)
    st.dataframe(results_df)

# Model evaluation section below both columns
st.subheader("Model Performance")
st.write(f"**MAE GBR:** {mean_absolute_error(y_test, gbr_pred):.2f}")
st.write(f"**MAE ARIMA:** {mean_absolute_error(y_test, arima_pred):.2f}")
st.write(f"**MAE Holt:** {mean_absolute_error(y_test, holt_pred):.2f}")
st.write(f"**GBR R²:** {gbr.score(X_train, y_train):.2f}")
