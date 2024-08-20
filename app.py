import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from predict_model import load_model_and_scalers, predict, save_plots, feature_engineering

def main():
    st.sidebar.title("Stock Market Analysis and Prediction")

    base_directory = os.path.dirname(os.path.abspath(__file__))
    tickers_file = os.path.join(base_directory, 'tickers.txt')
    with open(tickers_file, 'r') as file:
        tickers = file.read().splitlines()

    selected_ticker = st.sidebar.selectbox("Select stock ticker", tickers)
    future_days = st.sidebar.slider("Number of days to predict", 1, 360, 30)

    st.title(f"{selected_ticker} Stocks")
    data_path = os.path.join(base_directory, f'data/{selected_ticker}/{selected_ticker}.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    st.subheader(f"Recent {selected_ticker} Stock Data")
    st.dataframe(df.tail(10))

    st.subheader(f"{selected_ticker} Stock Prices")
    fig = px.line(df, x=df.index, y='Close', title=f'{selected_ticker} Live Share Price')
    st.plotly_chart(fig)

    if st.sidebar.button("Predict"):
        model = load_model_and_scalers(selected_ticker, os.path.join(base_directory, 'models'))

        if model:
            df_processed = pd.read_csv(os.path.join(base_directory, f'preprocessed_data/{selected_ticker}/{selected_ticker}.csv'))
            df_processed = feature_engineering(df_processed)

            forecast = predict(model, future_days, df_processed)

            forecast = forecast.tail(future_days)

            prediction_dates = pd.date_range(df.index[-1], periods=future_days + 1).tolist()

            prediction_df = pd.DataFrame({
                "Date": prediction_dates[1:], 
                "Predicted Price": forecast['yhat'][-future_days:]
            })

            prediction_df.index = range(1, len(prediction_df) + 1)

            save_plots(model, forecast, selected_ticker, base_directory)

            st.subheader(f"Stock Price Prediction for the next {future_days} days")
            st.write(prediction_df)

            fig = px.line(prediction_df, x='Date', y='Predicted Price', title=f'{selected_ticker} Price Prediction')
            st.plotly_chart(fig)
        else:
            st.error("Model not found or failed to load. Please check the logs.")

if __name__ == "__main__":
    main()
