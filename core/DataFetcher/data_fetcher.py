import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcherError(Exception):
    """Custom exception for DataFetcher errors."""
    pass

class DataFetcher:
    """Class to fetch financial data using yfinance
        Example:
            fetcher = DataFetcher()
            data = fetcher.fetch_data(ticker="AAPL")"""
    def __init__(self, ticker_symbol: str):
        """Initialize DataFetcher with a ticker symbol."""
        if not ticker_symbol or not isinstance(ticker_symbol, str):
            raise DataFetcherError("Invalid ticker symbol provided.")
        
        self.ticker_symbol = ticker_symbol.upper().strip()
        self.ticker = yf.Ticker(self.ticker_symbol)
        self._cache={}

    def get_info(self) -> Dict[str, Any]:
        """Fetch general information about the ticker."""
        if 'info' in self._cache:
            return self._cache['info']
        try:
            info = self.ticker.info
            if not info or 'symbol' not in info:
                raise DataFetcherError(f"No information found for ticker: {self.ticker_symbol}")
            self._cache['info'] = info
            return info
        except Exception as e:
            raise DataFetcherError(f"Error fetching info for ticker {self.ticker_symbol}: {str(e)}")
    
    def get_income_statement(self) -> pd.DataFrame:
        """Fetch the income statement data."""
        if 'income_stmt' in self._cache:
            return self._cache['income_stmt']
        try:
            income_stmt = self.ticker.financials
            if income_stmt is None or income_stmt.empty:
                raise DataFetcherError(f"No income statement data found for ticker: {self.ticker_symbol}")
            self._cache['income_stmt'] = income_stmt
            return income_stmt
        except Exception as e:
            raise DataFetcherError(f"Error fetching income statement for ticker {self.ticker_symbol}: {str(e)}")
        
    def get_balance_sheet(self) -> pd.DataFrame:
        """Fetch the balance sheet data."""
        if 'balance_sheet' in self._cache:
            return self._cache['balance_sheet']
        try:
            balance_sheet = self.ticker.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                raise DataFetcherError(f"No balance sheet data found for ticker: {self.ticker_symbol}")
            self._cache['balance_sheet'] = balance_sheet
            return balance_sheet
        except Exception as e:
            raise DataFetcherError(f"Error fetching balance sheet for ticker {self.ticker_symbol}: {str(e)}")
    
    def get_cash_flow(self) -> pd.DataFrame:
        """Fetch the cash flow data."""
        if 'cash_flow' in self._cache:
            return self._cache['cash_flow']
        try:
            cash_flow = self.ticker.cashflow
            if cash_flow is None or cash_flow.empty:
                raise DataFetcherError(f"No cash flow data found for ticker: {self.ticker_symbol}")
            self._cache['cash_flow'] = cash_flow
            return cash_flow
        except Exception as e:
            raise DataFetcherError(f"Error fetching cash flow for ticker {self.ticker_symbol}: {str(e)}")
    
    def get_share_price_data(self, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical share price data."""
        cache_key = f"share_price_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            share_price_data = self.ticker.history(period=period)
            if share_price_data.empty:
                raise DataFetcherError(f"No share price data found for ticker: {self.ticker_symbol} with period: {period}")
            share_price_data.reset_index(inplace=True)
            self._cache[cache_key] = share_price_data
            return share_price_data
        except Exception as e:
            raise DataFetcherError(f"Error fetching share price data for ticker {self.ticker_symbol}: {str(e)}")
        
    def get_dividends(self) -> pd.DataFrame:
        """Fetch dividend data."""
        if 'dividends' in self._cache:
            return self._cache['dividends']
        try:
            dividends = self.ticker.dividends
            self._cache['dividends'] = dividends
            return dividends
        except Exception as e:
            logger.warning(f"Failed to fetch dividends for {self.ticker_symbol}: {str(e)}")
            return pd.Series()
