import os
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)

def load_tickers(filename):
    try:
        with open(filename, 'r') as file:
            tickers = file.read().splitlines()
        logging.info(f"Loaded tickers: {tickers}")
        return tickers
    except Exception as e:
        logging.error(f"Error loading tickers from file {filename}: {e}")
        return []

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.error(f"Missing columns for {ticker}: {missing_columns}")
        return pd.DataFrame()
    
    return data[required_columns]

def save_stock_data(ticker, data):
    if data.empty:
        logging.warning(f"No data to save for {ticker}.")
        return
    
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(base_directory, f'data/{ticker}')

    os.makedirs(data_directory, exist_ok=True)

    data.to_csv(os.path.join(data_directory, f'{ticker}.csv'))

    logging.info(f"Saved data for {ticker}")

def main():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_directory, 'tickers.txt')
    tickers = load_tickers(file_path)

    for ticker in tickers:
        data = fetch_stock_data(ticker)
        save_stock_data(ticker, data)

if __name__ == "__main__":
    main()
