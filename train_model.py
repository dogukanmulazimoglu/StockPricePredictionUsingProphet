from datetime import datetime
import os
import logging
import numpy as np
import pandas as pd
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from collect_data import load_tickers
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def load_preprocessed_data(ticker, base_directory):
    preprocessed_data_path = os.path.join(base_directory, f'preprocessed_data/{ticker}/{ticker}.csv')
    df = pd.read_csv(preprocessed_data_path)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert(None)
    return df

def save_prophet_plots(model, forecast, ticker, base_directory):
    plot_directory = os.path.join(base_directory, 'train_plots', ticker)
    os.makedirs(plot_directory, exist_ok=True)
    
    fig1 = plt.figure()
    model.plot(forecast)
    plt.title(f"{ticker} Prophet Model Forecast")
    plt.savefig(os.path.join(plot_directory, f'{ticker}_prophet_forecast.png'))
    plt.close(fig1)
    
    fig2 = plt.figure()
    model.plot_components(forecast)
    plt.savefig(os.path.join(plot_directory, f'{ticker}_prophet_components.png'))
    plt.close(fig2)
    
    logging.info(f"Prophet plots saved for {ticker}.")

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mape

def save_model_performance(ticker, best_params, mse, rmse, mape, base_directory):
    performance_path = os.path.join(base_directory, 'model_performance.csv')
    model_performance = {
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Ticker': ticker,
        'Best Params': best_params,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
    if not os.path.exists(performance_path):
        df_performance = pd.DataFrame([model_performance])
    else:
        df_performance = pd.read_csv(performance_path)
        df_performance = pd.concat([df_performance, pd.DataFrame([model_performance])], ignore_index=True)
    df_performance.to_csv(performance_path, index=False)
    logging.info(f"Model performance for {ticker} saved to {performance_path}.")

def generate_holidays(start_year, end_year):
    holidays = []
    for year in range(start_year, end_year + 1):
        holidays.extend([
            {'holiday': 'New Year', 'ds': pd.to_datetime(f'{year}-01-01'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Christmas', 'ds': pd.to_datetime(f'{year}-12-25'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Thanksgiving', 'ds': pd.to_datetime(f'{year}-11-23'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Independence Day', 'ds': pd.to_datetime(f'{year}-07-04'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Labor Day', 'ds': pd.to_datetime(f'{year}-09-04'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Earnings Report', 'ds': pd.to_datetime(f'{year}-10-20'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Black Friday', 'ds': pd.to_datetime(f'{year}-11-24'), 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Cyber Monday', 'ds': pd.to_datetime(f'{year}-11-27'), 'lower_window': 0, 'upper_window': 1},
        ])
    return pd.DataFrame(holidays)

def objective(params, df_train, df_test, best_score, patience):
    params['seasonality_mode'] = 'additive' if params['seasonality_mode'] == 0 else 'multiplicative'
    model = Prophet(**params)
    model.add_regressor('lag_1')
    model.add_regressor('lag_7')
    model.add_regressor('rsi')
    model.add_regressor('macd')
    model.add_regressor('macd_diff')
    model.fit(df_train)
    
    forecast = model.predict(df_test)
    
    mse, rmse, mape = evaluate_model(df_test['y'], forecast['yhat'])
    logging.info(f"Evaluating parameters {params}: MSE = {mse}, RMSE = {rmse}, MAPE = {mape}")
    
    if mse < best_score:
        best_score = mse
        patience = 0
    else:
        patience += 1
    
    return mse, best_score, patience

def bayesian_optimization(df_train, df_test):
    space = {
        'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
        'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
        'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 10.0)
    }
    
    trials = Trials()
    best_score = float('inf')
    patience = 0
    max_patience = 10
    max_evals = 100
    
    for i in range(max_evals):
        best = fmin(fn=lambda params: objective(params, df_train, df_test, best_score, patience)[0],
                    space=space,
                    algo=tpe.suggest,
                    max_evals=i+1,
                    trials=trials)
        score, best_score, patience = objective(best, df_train, df_test, best_score, patience)
        if patience >= max_patience:
            logging.info(f"Early stopping triggered after {i+1} iterations.")
            break
    
    logging.info(f"Best parameters found: {best}")
    return best

def train_prophet_model(df, ticker, base_directory):
    start_year = df['ds'].min().year
    end_year = df['ds'].max().year
    holidays = generate_holidays(start_year, end_year)
    
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    
    logging.info(f"Starting Bayesian optimization for {ticker}...")
    best_params = bayesian_optimization(df_train, df_test)
    
    best_params['seasonality_mode'] = 'additive' if best_params['seasonality_mode'] == 0 else 'multiplicative'
    
    model = Prophet(holidays=holidays, **best_params)
    model.add_regressor('lag_1')
    model.add_regressor('lag_7')
    model.add_regressor('rsi')
    model.add_regressor('macd')
    model.add_regressor('macd_diff')
    model.fit(df_train)
    
    model_path = os.path.join(base_directory, f'models/{ticker}/{ticker}_prophet_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Best model for {ticker} saved with parameters: {best_params}")
    
    forecast = model.predict(df_test)
    
    save_prophet_plots(model, forecast, ticker, base_directory)
    
    mse, rmse, mape = evaluate_model(df_test['y'], forecast['yhat'])
    save_model_performance(ticker, best_params, mse, rmse, mape, base_directory)
    
    return model

def train_all_models():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_directory, 'tickers.txt')
    tickers = load_tickers(file_path)
    
    for ticker in tickers:
        logging.info(f"Training model for {ticker}...")
        df = load_preprocessed_data(ticker, base_directory)
        if not df.empty:
            train_prophet_model(df, ticker, base_directory)
        else:
            logging.warning(f"No data available for {ticker}. Skipping...")

if __name__ == "__main__":
    train_all_models()
