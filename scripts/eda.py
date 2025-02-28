import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm

class ExplanatoryAnalysis():
    def __init__(self):
        pass
    def to_time(self,df,name):
        self.name = name
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace= True)
        return df
    def rolling(self,df):
        """
        takes dataframe and reurn a dictionary containing "rolling_mean" and "rolling_std"
        """
        dic = {}
        self.returns = df.pct_change()
        # Calculate rolling mean and rolling standard deviation (volatility)
        self.rolling_mean = df.rolling(window=30).mean()  # 30-day moving average
        self.rolling_std = df.rolling(window=30).std()  # 30-day rolling standard deviation
        dic['rolling_mean'] = self.rolling_mean
        dic['rolling_std'] = self.rolling_std
        # Identify anomalies (outliers) based on Z-score
        z_scores = (self.returns - self.returns.mean()) / self.returns.std()
        self.outliers = self.returns[(z_scores.abs() > 3)]  # Consider Z-score > 3 as outlier

        # Find days with unusually high/low returns
        self.high_returns = self.returns[self.returns > self.returns.quantile(0.99)]  # Top 1% returns
        self.low_returns = self.returns[self.returns < self.returns.quantile(0.01)]  # Bottom 1% returns

        return dic
    def plot_closing_price(self, df):
        """

        """       
        # 1️⃣ Plot Closing Prices Over Time
        self.tickers = df.columns
        plt.figure(figsize=(12, 5))
        for ticker in self.tickers:
            plt.plot(df.index, df[ticker], label=ticker)
        plt.title(f"{self.name} Closing Prices Over Time")
        plt.xlabel("Year")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()
        return plt
    
    def plot_daily_percentage_change(self,df):
        # 2️⃣ Plot Daily Percentage Change (Volatility)
        plt.figure(figsize=(12, 5))
        for ticker in self.tickers:
            plt.plot(self.returns.index, self.returns[ticker], label=ticker, alpha=0.7)
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title(f"{self.name} Daily Percentage Change (Volatility)")
        plt.xlabel("Year")
        plt.ylabel("Daily Returns")
        plt.legend()
        plt.grid()
        return plt
    
    def plot_rolling_mean(self, df):
        # 3️⃣ Plot Rolling Mean and Rolling Standard Deviation
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        for ticker in self.tickers:
            ax[0].plot(df.index, self.rolling_mean[ticker], label=ticker)
        ax[0].set_title(f"{self.name} 30-Day Moving Average")
        ax[0].legend()
        ax[0].grid()

        for ticker in self.tickers:
            ax[1].plot(df.index, self.rolling_std[ticker], label=ticker, alpha=0.7)
        ax[1].set_title(f"{self.name} 30-Day Rolling Standard Deviation (Volatility)")
        ax[1].legend()
        ax[1].grid()

        return plt
    def box_plot(self, df):
        # 4️⃣ Outlier Detection - Highlight Extreme Daily Returns
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=self.returns, showfliers=True)
        plt.title(f"{self.name} Outlier Detection in Daily Returns")
        plt.ylabel("Daily Percentage Change")
        plt.xticks(rotation=45)
        plt.grid()

        return plt
    
    def plot_trend(self,df):
        # Decompose the time series (using an additive model)
        decomposition = seasonal_decompose(df.iloc[:,1], model="additive", period=252)  # 252 trading days in a year

        # Plot decomposition results
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        df.iloc[:,1].plot(ax=axes[0], title=f"{self.name} Stock Price - Observed", color="blue")
        decomposition.trend.plot(ax=axes[1], title="Trend", color="red")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonality", color="green")
        decomposition.resid.plot(ax=axes[3], title="Residual (Noise)", color="gray")

        plt.tight_layout()
        return plt
    def value_at_risk(self, df):
        # Calculate daily VaR at 95% confidence level
        df['returns'] = df.iloc[:,1].pct_change()
        confidence_level = 0.05
        VaR_95 = df['returns'].quantile(confidence_level)
        print(f"{self.name}'s Value at Risk (95% confidence level): {VaR_95:.4f}")

        # Calculate Sharpe Ratio (assuming risk-free rate = 0%)
        risk_free_rate = 0
        sharpe_ratio = (df["returns"].mean() - risk_free_rate) / df["returns"].std()
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

        # Plot histogram of daily returns
        plt.figure(figsize=(12, 5))
        sns.histplot(df['returns'], bins=50, kde=True, color="purple")
        plt.axvline(VaR_95, color="red", linestyle="dashed", label=f"VaR (95%) = {VaR_95:.4f}")
        plt.title(f"{self.name} Daily Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.legend()
        return plt

