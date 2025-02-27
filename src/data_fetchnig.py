import pandas as pd 
import yfinance as yf

tickers = ['TSLA', 'BND', 'SPY']

start_date = '2015-01-01'
end_date = '2025-01-31'

data = yf.download(tickers,start_date,end_date)
data.columns = ["{}_{}".format(col[0], col[1]) for col in data.columns]

data.to_csv("financial_raw_data.csv")