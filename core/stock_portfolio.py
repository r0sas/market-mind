from core.stock import Stock  # assume you have this somewhere
from core.Ticker import Ticker
import pandas as pd

class Stock_portfolio:
    def __init__(self):
        pass
    
    def get_historical_rentability(self, ticker_symbol, df):
        print(ticker_symbol)
        temp_df = df[df["Ticker"] == ticker_symbol].copy()

        # Get earliest acquisition date in Unix time
        min_date = int(temp_df["Acquisition Date"].min().timestamp())

        # Fetch close prices only once per ticker
        ticker = Ticker(ticker_symbol)
        close_df = ticker.get_close_price(min_date)
        print(close_df)
        close_df['Date'] = pd.to_datetime(close_df['Date']).dt.normalize()
        close_df['Invested Amount'] = 0.0
        close_df['Acq Shares'] = 0.0

        # Apply acquisitions to the corresponding dates
        for _, row in temp_df.iterrows():
            acquisition_date = row['Acquisition Date']
            close_df.loc[close_df["Date"] == acquisition_date, 'Invested Amount'] += row["Investment ($)"]
            close_df.loc[close_df["Date"] == acquisition_date, 'Acq Shares'] += row['Shares']

        # Compute cumulative totals
        close_df['Shares'] = close_df['Acq Shares'].cumsum()
        close_df['Total Invested'] = close_df['Invested Amount'].cumsum()
        close_df['Current Value'] = close_df['Close'] * close_df['Shares']
        close_df['ROI'] = ((close_df['Current Value'] - close_df['Total Invested']) / close_df['Total Invested']) * 100
        close_df['Ticker'] = ticker_symbol

        return close_df


    def get_portfolio_historical_rentability(self, df):
        df["Acquisition Date"] = pd.to_datetime(df["Acquisition Date"])
        df_list = []

        for ticker_symbol in df["Ticker"].unique():
            close_df = self.get_historical_rentability(ticker_symbol, df)
            df_list.append(close_df)

        # Concatenate all ticker data
        result_df = pd.concat(df_list, ignore_index=True)

        # Aggregate portfolio by date
        agg_df = result_df.groupby('Date').agg({
            'Current Value': 'sum',
            'Total Invested': 'sum'
        }).reset_index()
        agg_df['ROI'] = ((agg_df['Current Value'] - agg_df['Total Invested']) / agg_df['Total Invested']) * 100

        return agg_df, result_df
    
    def get_historical_data(self, selected_stocks):
        df = pd.DataFrame(columns=[
        "Date", "Close", "Ticker"])
        for symbol in selected_stocks:
            ticker = Ticker(symbol)
            hist_data = ticker.get_close_price()
            hist_data["Ticker"] = symbol
            hist_data['Rentability'] = hist_data['Close'].pct_change()
            hist_data['Cumulative Rentability'] = ((1 + hist_data['Rentability']).cumprod() - 1)
            #hist_data['Rentability'] = hist_data['Rentability']*100
            #hist_data['Cumulative Rentability'] = hist_data['Cumulative Rentability']*100
            df = pd.concat([df, hist_data], ignore_index=True)
        return df