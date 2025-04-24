import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import requests
import os, sys

# Custom module paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_lstm import build_lstm_model
from model_xgb import build_xgb_model
from data_loader import download_stock_data
from preprocessing import preprocess_data

# MUST BE FIRST
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

# Tab 1: Home
with tab1:
    st.title("📈 Welcome to Stock Price Predictor")
    st.markdown("""
    This app lets you:
    - Predict stock prices using **LSTM** or **XGBoost**
    - Forecast future stock prices
    - View real-time **news** about the selected stock
    """)

# Tab 2: Model Prediction
with tab2:
    st.header("📊 Model Prediction (Train-Test)")
    if st.button("Run Prediction"):
        df = download_stock_data(ticker, str(start_date), str(end_date))
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df, seq_length)

        if model_type == "LSTM":
            with st.spinner("🧠 Training LSTM model..."):
                model = build_lstm_model((seq_length, 1))
                X_train = X_train.reshape((X_train.shape[0], seq_length, 1))
                X_test = X_test.reshape((X_test.shape[0], seq_length, 1))
                model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
                predictions = model.predict(X_test)
        else:
            with st.spinner("🧠 Training XGBoost model..."):
                model = build_xgb_model()
                X_train = X_train.reshape((X_train.shape[0], seq_length))
                X_test = X_test.reshape((X_test.shape[0], seq_length))
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

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

# Tab 3: Future Forecast
with tab3:
    st.header("⏳ Future Stock Price Forecast")
    df = download_stock_data(ticker, str(start_date), str(end_date))
    _, _, _, _, scaler = preprocess_data(df, seq_length)
    model = build_lstm_model((seq_length, 1))

    all_data = df['Close'].values.reshape(-1, 1)
    all_scaled = scaler.fit_transform(all_data)

    X = []
    for i in range(seq_length, len(all_scaled)):
        X.append(all_scaled[i - seq_length:i])
    X = np.array(X)

    model.fit(X, all_scaled[seq_length:], epochs=epochs, batch_size=32, verbose=0)

    last_seq = all_scaled[-seq_length:]
    predictions = []

    for _ in range(forecast_days):
        pred = model.predict(last_seq.reshape(1, seq_length, 1))[0][0]
        predictions.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    st.subheader(f"📆 Predicted Stock Prices for Next {forecast_days} Day(s)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(future_prices, label="Future Prediction", color="orange")
    ax.set_xlabel("Days from Today")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    st.pyplot(fig)

# Tab 4: News Updates
with tab4:
    st.header(f"📰 Latest News for {ticker}")
    api_key = "7f14023dd95d45d884e821f2e44fcfb5"
    news_url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={api_key}"

    try:
        response = requests.get(news_url)
        news_data = response.json()
        if news_data["status"] == "ok":
            articles = news_data["articles"][:5]
            for article in articles:
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.write(article["description"])
                st.caption(f"🕒 {article['publishedAt']}")
        else:
            st.warning("No news found.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
