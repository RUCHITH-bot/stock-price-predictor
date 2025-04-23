import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_lstm import build_lstm_model
from model_xgb import build_xgb_model
from data_loader import download_stock_data, get_stock_news
from preprocessing import preprocess_data
from indicators import add_indicators

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

st.title("📊 AI-Based Stock Insights App")

tab1, tab2 = st.tabs(["📈 Price Prediction", "📰 Stock News"])

with tab1:
    st.header("Predict Stock Prices")

    ticker = st.sidebar.text_input("Enter Ticker", value="AAPL")
    start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    model_type = st.sidebar.selectbox("Model", ["LSTM", "XGBoost"])
    seq_length = st.sidebar.slider("Sequence Length", 30, 100, 60)
    epochs = st.sidebar.slider("Epochs (LSTM)", 1, 20, 10)

    if st.button("Run Prediction"):
        df = download_stock_data(ticker, str(start), str(end))
        df = add_indicators(df)

        st.subheader("📊 Price + Indicators")
        st.line_chart(df[["Close", "SMA_50", "RSI", "MACD"]])

        X_train, y_train, X_test, y_test, scaler = preprocess_data(df, seq_length)

        if model_type == "LSTM":
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            model = build_lstm_model((seq_length, 1))
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            predictions = model.predict(X_test)
        else:
            model = build_xgb_model()
            model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1])), y_train)
            predictions = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1])))

        predicted = scaler.inverse_transform(predictions.reshape(-1, 1))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        st.success(f"✅ RMSE: {rmse:.2f}")

        st.line_chart(pd.DataFrame({
            "Actual": actual.flatten(),
            "Predicted": predicted.flatten()
        }))

with tab2:
    st.header("Latest News for Selected Stock")
    news = get_stock_news(ticker)
    if news:
        for item in news:
            st.markdown(f"**[{item['title']}]({item['url']})**")
            st.caption(f"{item['publishedAt']}")
            st.write(item['description'])
    else:
        st.warning("No recent news found for this ticker.")
