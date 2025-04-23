import sys, os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Append the src folder to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import functions from src
from model_xgb import build_xgb_model
from data_loader import download_stock_data
from preprocessing import preprocess_data
from model_lstm import build_lstm_model

# Optional: Auto refresh every 30 seconds
st_autorefresh(interval=30 * 1000, key="refresh")

# Function to get live stock price
def get_live_price(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        todays_data = ticker.history(period='1d')
        if not todays_data.empty:
            current_price = todays_data['Close'].iloc[-1]
            return round(current_price, 2)
        return None
    except:
        return None

# Streamlit App Configuration
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Predictor")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
seq_length = st.sidebar.slider("Sequence Length", 30, 100, 60)
epochs = st.sidebar.slider("Epochs (for LSTM)", 1, 20, 10)
model_type = st.sidebar.selectbox("Select Model", ["LSTM", "XGBoost"])

# Display live stock price
st.subheader(f"📍 Live Stock Price for **{ticker}**")
live_price = get_live_price(ticker)
if live_price:
    st.metric(label="Current Price", value=f"${live_price}")
else:
    st.warning("Unable to fetch live price.")

if st.button("Predict"):

    with st.spinner("📥 Fetching data..."):
        df = download_stock_data(ticker, str(start_date), str(end_date))

    with st.spinner("🔄 Preprocessing data..."):
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df, seq_length)

    if model_type == "LSTM":
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        with st.spinner("🧠 Training LSTM model..."):
            model = build_lstm_model((seq_length, 1))
            model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=32, verbose=0)

        with st.spinner("📊 Predicting with LSTM..."):
            predictions = model.predict(X_test_lstm)
    else:
        X_train_flat = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[1]))

        with st.spinner("🧠 Training XGBoost model..."):
            model = build_xgb_model()
            model.fit(X_train_flat, y_train)

        with st.spinner("📊 Predicting with XGBoost..."):
            predictions = model.predict(X_test_flat)

    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    st.success(f"✅ RMSE: {rmse:.2f}")

    st.subheader("📉 Predicted vs Actual Closing Prices")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_prices, label="Actual")
    ax.plot(predicted_prices, label="Predicted")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
