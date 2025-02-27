import pandas as pd 
import yfinance as yf

tickers = ['TSLA', 'BND', 'SPY']

start_date = '2015-01-01'
end_date = '2025-01-31'
for ticker in tickers:
    data = yf.download(ticker,start_date,end_date)
    data.columns = ["{}_{}".format(col[0], col[1]) for col in data.columns]

    data.to_csv(f"{ticker}_raw_data.csv")