import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import requests
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_lstm import build_lstm_model
from model_xgb import build_xgb_model
from data_loader import download_stock_data
from preprocessing import preprocess_data

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.sidebar.title("🔍 Stock Predictor Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
seq_length = st.sidebar.slider("Sequence Length", 30, 100, 60)
epochs = st.sidebar.slider("Epochs (LSTM)", 1, 20, 10)
forecast_days = st.sidebar.slider("Days to Predict in Future", 1, 30, 7)
model_type = st.sidebar.selectbox("Model", ["LSTM", "XGBoost"])

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Model Prediction", "⏳ Future Forecast", "📰 News Updates"])

# Tab 1
with tab1:
    st.title("📈 Welcome to Stock Price Predictor")
    st.markdown("""
    Predict stock prices using **LSTM/XGBoost**, forecast future prices, and view latest stock news.
    """)

# Tab 2
with tab2:
    st.header("📊 Model Prediction")
    if st.button("Run Prediction"):
        df = download_stock_data(ticker, str(start_date), str(end_date))
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df, seq_length)

        if model_type == "LSTM":
            X_train_lstm = X_train.reshape((X_train.shape[0], seq_length, 1))
            X_test_lstm = X_test.reshape((X_test.shape[0], seq_length, 1))
            model = build_lstm_model((seq_length, 1))
            model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=32, verbose=0)
            predictions = model.predict(X_test_lstm)
        else:
            X_train_flat = X_train.reshape((X_train.shape[0], seq_length))
            X_test_flat = X_test.reshape((X_test.shape[0], seq_length))
            model = build_xgb_model()
            model.fit(X_train_flat, y_train)
            predictions = model.predict(X_test_flat)

        predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

        st.success(f"✅ RMSE: {rmse:.2f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(actual_prices, label="Actual")
        ax.plot(predicted_prices, label="Predicted")
        ax.set_title("Predicted vs Actual Prices")
        ax.legend()
        st.pyplot(fig)

# Tab 3
with tab3:
    st.header("⏳ Future Forecast")
    df = download_stock_data(ticker, str(start_date), str(end_date))
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    model = build_lstm_model((seq_length, 1))
    X = []
    y = []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length:i])
        y.append(scaled[i])
    X = np.array(X)
    y = np.array(y)

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    last_seq = scaled[-seq_length:]
    predictions = []
    current_input = last_seq.copy()

    for _ in range(forecast_days):
        pred = model.predict(current_input.reshape(1, seq_length, 1))[0][0]
        predictions.append(pred)
        current_input = np.append(current_input[1:], [[pred]], axis=0)

    future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    st.subheader(f"📆 Next {forecast_days} Days Forecast")
    fig, ax = plt.subplots()
    ax.plot(future_prices, label="Predicted")
    ax.set_ylabel("Price")
    ax.set_xlabel("Days")
    ax.legend()
    st.pyplot(fig)

# Tab 4
with tab4:
    st.header(f"📰 News for {ticker}")
    api_key = "7f14023dd95d45d884e821f2e44fcfb5"  # Your NewsAPI Key
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={api_key}"

    try:
        res = requests.get(url)
        articles = res.json().get("articles", [])[:5]
        if not articles:
            st.info("No recent news found.")
        for article in articles:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(article.get("description", "No description"))
            st.caption(f"🕒 {article['publishedAt']}")
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
