from yahooquery import Ticker
import pandas as pd
from typing import Optional, List

class FetchStockData:

    def __init__(self):
        pass

    def get_historical_data(self, symbols: List[str], start_date: Optional[str] = None, period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.

        Args:
            symbols: List of ticker symbols.
            start_date: e.g. '2020-01-01'
            period: e.g. '1y', 'max', '5d'
        """
        tickers = Ticker(symbols)
        kwargs = {}
        if start_date:
            kwargs['start'] = start_date
        elif period:
            kwargs['period'] = period
        else:
            kwargs['period'] = 'max'

        df = tickers.history(**kwargs).reset_index()
        df = df.drop(["open", "high", "low", "adjclose"], axis=1)        # can be useful for candlestick charts
        df = df.rename(columns={"symbol": "Ticker", "date": "Date"})
        df["Date"] = (pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize())
        if "dividends" not in df.columns.to_list():
            df["dividends"] = 0.0
        return df

    def get_current_price(self, symbols: list) -> dict:
        tickers = Ticker(symbols)
        price_data = tickers.price
        last_prices = {}
        for symbol in symbols:
            info = price_data[symbol]
            # 'regularMarketPrice' works for current or last close
            last_prices[symbol] = info.get('regularMarketPrice')
        return last_prices
    
    def get_stock_info(self, symbols: list[str]) -> dict:
        tickers = Ticker(symbols)
        info_data = tickers.summary_profile

        # build dictionary of industry and sector
        sectors_industries = {
            symbol: {
                "industry": info_data.get(symbol, {}).get("industry", "N/A"),
                "sector": info_data.get(symbol, {}).get("sector", "N/A")
            }
            for symbol in symbols
        }

        return sectors_industries