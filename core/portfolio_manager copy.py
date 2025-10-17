from core.fetch_stock_data import FetchStockData
import pandas as pd
from typing import Optional, List

class PortfolioManager:
    def __init__(self):
        self.data_fetcher = FetchStockData()
    
    def get_stocks_current_price(self, tickers: list) -> dict:
        return self.data_fetcher.get_current_price(tickers)

    def get_stocks_history(self, unique_symbols: List[str], start_date: Optional[str] = None, period: Optional[str] = None):
        if start_date and period:
            raise ValueError("Specify either 'start_date' or 'period', not both.")
        if start_date:
            return self.data_fetcher.get_historical_data(unique_symbols, start_date=start_date)
        elif period:
            return self.data_fetcher.get_historical_data(unique_symbols, period=period)
        else:
            return self.data_fetcher.get_historical_data(unique_symbols)

    # def get_portfolio_history(self, acq_df: pd.DataFrame) -> pd.DataFrame:
    #     min_date = acq_df["Date"].min()
    #     unique_symbols = acq_df["Ticker"].unique().tolist()
    #     historical_df = self.get_stocks_history(unique_symbols, start_date=min_date)
    #     df_merged = historical_df.merge(acq_df[["Ticker", "Date", "Shares", "Investment ($)"]], on=['Ticker', 'Date'], how='left')
    #     df_merged = df_merged.sort_values(['Ticker', 'Date'])
    #     df_merged['Investment ($)'] = df_merged['Investment ($)'].fillna(0.0)
    #     df_merged["Shares"] = df_merged["Shares"].fillna(0.0)
    #     df_merged["Shares"] = df_merged.groupby('Ticker')['Shares'].cumsum()
    #     df_merged["Investment ($)"] = df_merged.groupby('Ticker')['Investment ($)'].cumsum()
    #     df_merged["Current Value"] = df_merged['close'] * df_merged['Shares']
    #     df_merged['ROI'] = ((df_merged['Current Value'] - df_merged["Investment ($)"]) / df_merged["Investment ($)"]) * 100
    #     return df_merged
    
    def fetch_historical_prices(self, tickers: list, start_date: pd.Timestamp) -> pd.DataFrame:
        return self.get_stocks_history(tickers, start_date=start_date)

    def merge_acquisitions(self, historical_df: pd.DataFrame, acq_df: pd.DataFrame) -> pd.DataFrame:
        df_merged = historical_df.merge(
            acq_df[["Ticker", "Date", "Shares", "Investment ($)"]],
            on=['Ticker', 'Date'],
            how='left'
        )
        return df_merged.sort_values(['Ticker', 'Date'])

    def compute_cumulative(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Investment ($)'] = df['Investment ($)'].fillna(0.0)
        df['Shares'] = df['Shares'].fillna(0.0)
        df['Shares'] = df.groupby('Ticker')['Shares'].cumsum()
        df['Investment ($)'] = df.groupby('Ticker')['Investment ($)'].cumsum()
        return df

    def compute_pl_history(self, df: pd.Dataframe) -> pd.Dataframe:
        df["Profit/Loss"] = df["Current Value"] - df["Investment ($)"]
        return df

    def compute_roi(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Current Value'] = df['close'] * df['Shares']
        df['ROI'] = ((df['Current Value'] - df['Investment ($)']) / df['Investment ($)']) * 100
        return df    

    def get_portfolio_history(self, acq_df: pd.DataFrame) -> pd.DataFrame:
        min_date = acq_df["Date"].min()
        unique_symbols = acq_df["Ticker"].unique().tolist()
        
        historical_df = self.fetch_historical_prices(unique_symbols, min_date)
        df_merged = self.merge_acquisitions(historical_df, acq_df)
        df_merged = self.compute_cumulative(df_merged)
        df_merged = self.compute_roi(df_merged)
        
        return df_merged
    
    def get_stocks_rentability(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Rentability'] = df.groupby('Ticker')['close'].pct_change()
        df['Cumulative Rentability'] = (df.groupby('Ticker')['Rentability'].transform(lambda x: (1 + x).cumprod() - 1))
        return df
    
    def get_portfolio_rentability(self, df: pd.DataFrame):
        portfolio_df = df.groupby('Date').agg({
            'Current Value': 'sum',
            "Investment ($)": 'sum'
        }).reset_index()
        portfolio_df['ROI'] = ((portfolio_df['Current Value'] - portfolio_df["Investment ($)"]) / portfolio_df["Investment ($)"]) * 100
        portfolio_df["Profit/Loss"] = portfolio_df["Current Value"] - portfolio_df["Investment ($)"]
        return portfolio_df