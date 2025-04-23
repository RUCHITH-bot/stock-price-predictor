import yfinance as yf
import requests

def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

def get_stock_news(ticker):
    api_key = "7f14023dd95d45d884e821f2e44fcfb5"  # Keep this safe in real use
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        return data["articles"][:5]
    except:
        return []
