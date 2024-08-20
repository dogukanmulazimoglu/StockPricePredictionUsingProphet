import os
import logging
import pandas as pd
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD

logging.basicConfig(level=logging.INFO)

def load_model_and_scalers(ticker, model_dir):
    model_path = os.path.join(model_dir, f"{ticker}/{ticker}_prophet_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Loaded model for {ticker}.")
        return model
    else:
        logging.error(f"Model file not found for {ticker}.")
        return None

def feature_engineering(df):
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    df['rsi'] = RSIIndicator(close=df['y']).rsi()
    macd = MACD(close=df['y'])
    df['macd'] = macd.macd()
    df['macd_diff'] = macd.macd_diff()
    df.fillna(0, inplace=True)
    return df

def predict(model, future_days, df):
    future = model.make_future_dataframe(periods=future_days, freq='D')
    future = future[future['ds'].dt.weekday < 5]
    
    df_last = df.tail(len(future))
    
    if len(future) > len(df_last):
        future = future.iloc[-len(df_last):]

    future['lag_1'] = df_last['y'].shift(1).values
    future['lag_7'] = df_last['y'].shift(7).values
    future['rsi'] = RSIIndicator(close=df_last['y']).rsi().values
    future['macd'] = MACD(close=df_last['y']).macd().values
    future['macd_diff'] = MACD(close=df_last['y']).macd_diff().values

    future.fillna(0, inplace=True)
    
    forecast = model.predict(future)
    return forecast

def save_plots(model, forecast, ticker, base_directory):
    plot_directory = os.path.join(base_directory, 'predict_plots', ticker)
    os.makedirs(plot_directory, exist_ok=True)

    fig1 = plt.figure()
    model.plot(forecast)
    plt.title(f"{ticker} Stock Price Prediction")
    plt.savefig(os.path.join(plot_directory, f'{ticker}_forecast.png'))
    plt.close(fig1)

    fig2 = plt.figure()
    model.plot_components(forecast)
    plt.savefig(os.path.join(plot_directory, f'{ticker}_components.png'))
    plt.close(fig2)

    logging.info(f"Plots saved for {ticker}.")
