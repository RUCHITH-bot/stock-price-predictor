import streamlit as st
import time
import requests

# Set up your NewsAPI key (sign up at https://newsapi.org/)
API_KEY = '7f14023dd95d45d884e821f2e44fcfb5'  # Replace with your actual API key

# Placeholder for news updates
news_placeholder = st.empty()

# Streamlit app UI
st.set_page_config(page_title="Stock News Updates", layout="centered")
st.title("📰 Real-Time Stock News Updates")

# Input: User provides stock ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")

# Display the ticker in the main area
st.write(f"Latest News for: {ticker}")

# Function to fetch news related to the stock using NewsAPI
def fetch_stock_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}&pageSize=5'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles
    else:
        return []

# Real-time stock news updates
while True:
    with st.spinner("Fetching real-time stock news..."):
        try:
            articles = fetch_stock_news(ticker)
            if articles:
                # Clear the placeholder and show the latest news
                news_placeholder.empty()
                for article in articles:
                    title = article.get('title')
                    description = article.get('description')
                    url = article.get('url')

                    # Display the news title and description with a link to the full article
                    st.markdown(f"### {title}")
                    st.write(f"{description}")
                    st.markdown(f"[Read more]({url})")
                    st.write("---")  # Separator between articles
            else:
                news_placeholder.empty()
                st.write("❌ No news available at the moment.")
        except Exception as e:
            news_placeholder.empty()
            st.write(f"❌ Failed to fetch news. Error: {e}")
    
    # Wait for 30 seconds before the next update
    time.sleep(30)

    
import streamlit as st
import time
import requests

# Set the page configuration first (before any other Streamlit command)
st.set_page_config(page_title="Stock News Updates", layout="centered")

# Set up your NewsAPI key
API_KEY = '7f14023dd95d45d884e821f2e44fcfb5'  # Your actual API key

# Placeholder for news updates
news_placeholder = st.empty()

# Streamlit app UI
st.title("📰 Real-Time Stock News Updates")

# Input: User provides stock ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")

# Display the ticker in the main area
st.write(f"Latest News for: {ticker}")

# Function to fetch news related to the stock using NewsAPI
def fetch_stock_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}&pageSize=5'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles
    else:
        return []

# Real-time stock news updates
while True:
    with st.spinner("Fetching real-time stock news..."):
        try:
            articles = fetch_stock_news(ticker)
            if articles:
                # Clear the placeholder and show the latest news
                news_placeholder.empty()
                for article in articles:
                    title = article.get('title')
                    description = article.get('description')
                    url = article.get('url')

                    # Display the news title and description with a link to the full article
                    st.markdown(f"### {title}")
                    st.write(f"{description}")
                    st.markdown(f"[Read more]({url})")
                    st.write("---")  # Separator between articles
            else:
                news_placeholder.empty()
                st.write("❌ No news available at the moment.")
        except Exception as e:
            news_placeholder.empty()
            st.write(f"❌ Failed to fetch news. Error: {e}")
    
    # Wait for 30 seconds before the next update
    time.sleep(30)
