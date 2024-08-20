import os
import pandas as pd
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD

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

def feature_engineering(df):
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day_of_week'] = df['ds'].dt.dayofweek

    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    
    df['rsi'] = RSIIndicator(close=df['y']).rsi()
    macd = MACD(close=df['y'])
    df['macd'] = macd.macd()
    df['macd_diff'] = macd.macd_diff()

    df.fillna(0, inplace=True)

    return df

def load_stock_data(ticker):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_directory, f'data/{ticker}/{ticker}.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded data for {ticker}.")
        return df[['Close']].rename(columns={'Close': 'y'})
    else:
        logging.warning(f"Data for {ticker} not found.")
        return pd.DataFrame()

def save_preprocessed_data(ticker, df):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data_dir = os.path.join(base_directory, 'preprocessed_data')
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    ticker_dir = os.path.join(preprocessed_data_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    preprocessed_data_path = os.path.join(ticker_dir, f'{ticker}.csv')
    df.to_csv(preprocessed_data_path, index=False)
    logging.info(f"Saved preprocessed data for {ticker} in {preprocessed_data_path}.")

def preprocess_data():
    tickers = load_tickers('tickers.txt')
    for ticker in tickers:
        logging.info(f"Processing {ticker}...")
        df = load_stock_data(ticker)
        if not df.empty:
            df = df.reset_index()
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert(None)
            df = feature_engineering(df)
            save_preprocessed_data(ticker, df)
        else:
            logging.warning(f"No data found for {ticker}.")

if __name__ == "__main__":
    preprocess_data()
