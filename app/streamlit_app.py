import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import requests
from sklearn.metrics import mean_squared_error
from model_xgb import build_xgb_model
from model_lstm import build_lstm_model
from preprocessing import preprocess_data
from data_loader import download_stock_data

# Set this FIRST
st.set_page_config(page_title="📊 Stock Predictor & News", layout="wide")

# News API key
API_KEY = "7f14023dd95d45d884e821f2e44fcfb5"

# Sidebar Inputs
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
seq_length = st.sidebar.slider("Sequence Length", 30, 100, 60)
epochs = st.sidebar.slider("Epochs (LSTM)", 1, 20, 10)
model_type = st.sidebar.selectbox("Select Model", ["LSTM", "XGBoost"])

st.title("📈 Stock Price Predictor + 📰 News Updates")
st.markdown(f"### Selected Ticker: `{ticker}`")

# Predict button
if st.button("Predict"):
    with st.spinner("📥 Downloading data..."):
        df = download_stock_data(ticker, str(start_date), str(end_date))

    with st.spinner("🔄 Preprocessing..."):
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df, seq_length)

    if model_type == "LSTM":
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        with st.spinner("🧠 Training LSTM..."):
            model = build_lstm_model((seq_length, 1))
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        with st.spinner("📊 Predicting..."):
            predictions = model.predict(X_test)
    else:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        with st.spinner("🧠 Training XGBoost..."):
            model = build_xgb_model()
            model.fit(X_train, y_train)
        with st.spinner("📊 Predicting..."):
            predictions = model.predict(X_test)

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

# News Section
st.subheader("📰 Live News Feed")

def fetch_stock_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}&pageSize=5&sortBy=publishedAt"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get("articles", [])
    else:
        return []

articles = fetch_stock_news(ticker)
if articles:
    for article in articles:
        st.markdown(f"### {article['title']}")
        st.write(article['description'])
        st.markdown(f"[🔗 Read more]({article['url']})")
        st.write("---")
else:
    st.warning("No news available at the moment.")
