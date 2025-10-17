from core.fetch_stock_data import FetchStockData
import pandas as pd
from typing import Optional, List

class PortfolioManager:
    def __init__(self):
        self.data_fetcher = FetchStockData()

    # -------------------
    # Data Fetching
    # -------------------
    def get_stocks_current_price(self, tickers: list) -> dict:
        return self.data_fetcher.get_current_price(tickers)


    # def get_stocks_history(self, tickers: list, start_date: Optional[str] = None, period: Optional[str] = None):
    #     if start_date and period:
    #         raise ValueError("Specify either 'start_date' or 'period', not both.")
    #     return self.data_fetcher.get_historical_data(tickers, start_date=start_date, period=period)
    def get_stocks_history(self, unique_symbols: List[str], start_date: Optional[str] = None, period: Optional[str] = None):
        if start_date and period:
            raise ValueError("Specify either 'start_date' or 'period', not both.")
        if start_date:
            return self.data_fetcher.get_historical_data(unique_symbols, start_date=start_date)
        elif period:
            return self.data_fetcher.get_historical_data(unique_symbols, period=period)
        else:
            return self.data_fetcher.get_historical_data(unique_symbols)

    # -------------------
    # Portfolio Calculations
    # -------------------
    def get_portfolio_history(self, acq_df: pd.DataFrame) -> pd.DataFrame:
        min_date = acq_df["Date"].min()
        tickers = acq_df["Ticker"].unique().tolist()
        acq_df['Date'] = pd.to_datetime(acq_df['Date']).dt.normalize()
        historical_df = self.get_stocks_history(tickers, start_date=min_date)
        # Pipeline: merge → cumulative → ROI
        df = (self._merge_acquisitions(historical_df, acq_df)
                .pipe(self._compute_cumulative)
                .pipe(self._compute_current_value_and_roi)
                .pipe(self._compute_cumulative_dividends))
        return df

    def get_stocks_rentability(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Rentability'] = df.groupby('Ticker')['close'].pct_change()
        df['Cumulative Rentability'] = df.groupby('Ticker')['Rentability'].transform(lambda x: (1 + x).cumprod() - 1)*100
        return df

    def get_portfolio_rentability(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df)
        portfolio_df = df.groupby('Date').agg({
            'Current Value': 'sum',
            "Investment ($)": 'sum',
            "dividends": 'sum',
            "Cumulative Dividends": "sum"
        }).reset_index()
        portfolio_df["Profit/Loss"] = portfolio_df["Current Value"] - portfolio_df["Investment ($)"]
        portfolio_df['Rentability'] = portfolio_df['Current Value'].pct_change()
        portfolio_df['Cumulative Rentability'] = portfolio_df['Rentability'].transform(lambda x: (1 + x).cumprod() - 1)*100
        portfolio_df['ROI'] = (portfolio_df["Profit/Loss"] / portfolio_df["Investment ($)"]) * 100
        portfolio_df["close"] = 0.0
        portfolio_df["Ticker"] = "Portfolio"
        portfolio_df["Volume"] = None
        portfolio_df["Shares"] = None               # can probably remove on the dataframe used to plot
        return portfolio_df

    # -------------------
    # Internal Helper Methods
    # -------------------
    def _merge_acquisitions(self, historical_df: pd.DataFrame, acq_df: pd.DataFrame) -> pd.DataFrame:
        df = historical_df.merge(acq_df[["Ticker","Date","Shares","Investment ($)"]],
                                 on=['Ticker','Date'], how='left').sort_values(['Ticker','Date'])
        return df


    def _compute_cumulative_dividends(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Cumulative Dividends"] = df["Shares"] * df["dividends"]
        df["Cumulative Dividends"] = df.groupby('Ticker')["Cumulative Dividends"].cumsum()
        return df

    def _compute_cumulative(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Investment ($)'] = df['Investment ($)'].fillna(0.0)
        df['Shares'] = df['Shares'].fillna(0.0)
        df['Shares'] = df.groupby('Ticker')['Shares'].cumsum()
        df['Investment ($)'] = df.groupby('Ticker')['Investment ($)'].cumsum()
        return df

    def _compute_current_value_and_roi(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Current Value'] = df['close'] * df['Shares']
        df['Profit/Loss'] = df['Current Value'] - df['Investment ($)']
        df['ROI'] = (df['Profit/Loss'] / df['Investment ($)']) * 100
        return df
