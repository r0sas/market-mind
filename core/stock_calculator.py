"""
Portfolio Analysis System - Three-Layer Architecture (Refactored)

Layer 1: FetchStockData - Data Access Layer
Layer 2: StockCalculator - Calculation/Processing Layer  
Layer 3: PortfolioAnalyzer - Portfolio Operations Layer
"""

from core.fetch_stock_data import FetchStockData
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# LAYER 2: STOCK CALCULATOR - Individual Stock Calculations
# ==============================================================================

class StockCalculator:
    """
    Handles all stock-level calculations and data processing.
    
    This is a stateless service that performs calculations on stock data:
    - Fetching current and historical price data
    - Computing per-stock performance metrics (ROI, profit/loss)
    - Daily and cumulative rentability calculations
    - Dividend accumulation
    
    All methods are pure transformations on DataFrames.
    """
    
    def __init__(self, data_fetcher: Optional[FetchStockData] = None):
        """
        Initialize the StockCalculator with a data fetcher.
        
        Args:
            data_fetcher: Optional FetchStockData instance. Creates new one if not provided.
        """
        self._data_fetcher = data_fetcher or FetchStockData()
        self._stock_info_cache = {}
    
    # -------------------
    # Data Fetching
    # -------------------
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for multiple stock tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping tickers to current prices
            
        Raises:
            ValueError: If fetching prices fails
        """
        try:
            return self._data_fetcher.get_current_price(tickers)
        except Exception as e:
            logger.error(f"Failed to fetch current prices: {e}")
            raise ValueError(f"Failed to fetch current prices: {str(e)}")
    
    def get_historical_data(
        self, 
        tickers: List[str], 
        start_date: Optional[str] = None, 
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in format 'YYYY-MM-DD' (mutually exclusive with period)
            period: Period string like '1y', '6mo', '1mo' (mutually exclusive with start_date)
            
        Returns:
            DataFrame with historical OHLCV data indexed by date and ticker
            
        Raises:
            ValueError: If both start_date and period are specified, or if fetching fails
            
        Example:
            >>> calc = StockCalculator()
            >>> data = calc.get_historical_data(['AAPL', 'MSFT'], start_date='2024-01-01')
        """
        if start_date and period:
            raise ValueError("Specify either 'start_date' or 'period', not both.")
        
        try:
            if start_date:
                logger.info(f"Fetching historical data for {len(tickers)} tickers from {start_date}")
                return self._data_fetcher.get_historical_data(tickers, start_date=start_date)
            elif period:
                logger.info(f"Fetching historical data for {len(tickers)} tickers, period: {period}")
                return self._data_fetcher.get_historical_data(tickers, period=period)
            else:
                logger.info(f"Fetching historical data for {len(tickers)} tickers (default period)")
                return self._data_fetcher.get_historical_data(tickers)
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise ValueError(f"Failed to fetch historical data: {str(e)}")
    
    def get_stock_info(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fetch stock metadata (sector, industry, etc.) with caching.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary with structure: {ticker: {industry: x, sector: y}, ...}
            
        Raises:
            ValueError: If fetching stock info fails
        """
        # Check cache first
        uncached_tickers = [t for t in tickers if t not in self._stock_info_cache]
        
        if uncached_tickers:
            try:
                logger.info(f"Fetching stock info for {len(uncached_tickers)} tickers")
                new_info = self._data_fetcher.get_stock_info(uncached_tickers)
                self._stock_info_cache.update(new_info)
            except Exception as e:
                logger.error(f"Failed to fetch stock info: {e}")
                raise ValueError(f"Failed to fetch stock info: {str(e)}")
        
        return {ticker: self._stock_info_cache.get(ticker, {}) for ticker in tickers}
    
    def clear_cache(self) -> None:
        """Clear the stock info cache."""
        self._stock_info_cache.clear()
        logger.info("Stock info cache cleared")
    
    # -------------------
    # Stock Metrics Calculations
    # -------------------
    
    def calculate_investment_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic investment metrics for stock positions.
        
        Calculates:
        - Investment ($): Total amount invested (Shares × Purchase Price)
        - Current Value ($): Current portfolio value (Shares × Current Price)
        - ROI (%): Return on investment percentage
        
        Args:
            df: DataFrame with columns ['Shares', 'Price', 'Current Price']
            
        Returns:
            DataFrame with added metric columns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Shares', 'Price', 'Current Price']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        df["Investment ($)"] = df["Shares"] * df["Price"]
        df["Current Value ($)"] = df["Shares"] * df["Current Price"]
        
        # Vectorized ROI calculation
        mask = df["Investment ($)"] != 0
        df["ROI (%)"] = 0.0
        df.loc[mask, "ROI (%)"] = (
            (df.loc[mask, "Current Price"] - df.loc[mask, "Price"]) / 
            df.loc[mask, "Price"]
        ) * 100
        
        return df
    
    def calculate_position_value(self, shares: float, current_price: float) -> float:
        """
        Calculate current market value of a position.
        
        Args:
            shares: Number of shares held
            current_price: Current price per share
            
        Returns:
            Total market value of the position
        """
        return shares * current_price
    
    def calculate_roi(self, investment: float, current_value: float) -> float:
        """
        Calculate return on investment percentage.
        
        Args:
            investment: Initial investment amount
            current_value: Current value of investment
            
        Returns:
            ROI as a percentage
        """
        if investment == 0:
            return 0.0
        return ((current_value - investment) / investment) * 100
    
    def compute_rentability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily and cumulative rentability (percentage change) per stock.
        
        Rentability measures the daily percentage change in stock price.
        Cumulative Rentability shows the compound growth over time.
        
        Args:
            df: DataFrame with columns ['Ticker', 'close', 'Date']
            
        Returns:
            DataFrame with added 'Rentability' and 'Cumulative Rentability' columns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Ticker', 'close']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        df['Rentability'] = df.groupby('Ticker')['close'].pct_change()
        df['Cumulative Rentability'] = (
            df.groupby('Ticker')['Rentability']
            .transform(lambda x: ((1 + x).cumprod() - 1) * 100)
        )
        
        return df
    
    def compute_accumulated_dividends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative dividend income from held positions.
        
        Args:
            df: DataFrame with columns ['Ticker', 'Shares', 'dividends', 'Date']
            
        Returns:
            DataFrame with 'Cumulative Dividends' column
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Ticker', 'Shares', 'dividends']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        
        # Calculate per-date dividend earnings
        df['Cumulative Dividends'] = df['Shares'] * df['dividends']
        
        # Accumulate dividends over time per security
        df['Cumulative Dividends'] = df.groupby('Ticker')['Cumulative Dividends'].cumsum()
        
        return df
    
    def compute_cumulative_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative share holdings and total investment per position.
        
        Args:
            df: DataFrame with transaction records
            
        Returns:
            DataFrame with cumulative share count and cumulative investment amount
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Ticker', 'Shares', 'Investment ($)']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        
        # Fill NaN values (dates without new transactions) with zeros
        df['Investment ($)'] = df['Investment ($)'].fillna(0.0)
        df['Shares'] = df['Shares'].fillna(0.0)
        
        # Accumulate holdings and investment per ticker
        df['Shares'] = df.groupby('Ticker')['Shares'].cumsum()
        df['Investment ($)'] = df.groupby('Ticker')['Investment ($)'].cumsum()
        
        return df
    
    def compute_position_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate current market value, unrealized gains/losses, and returns for each position.
        
        Args:
            df: DataFrame with cumulative shares and current prices
            
        Returns:
            DataFrame with Current Value, Profit/Loss, and ROI (%) columns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Shares', 'close', 'Investment ($)']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        
        # Calculate current value
        df['Current Value'] = df['close'] * df['Shares']
        
        # Add profit/loss and ROI using dedicated functions
        df = (df.pipe(self.add_profit_loss_column)
                .pipe(self.add_roi_column))
        
        return df
    
    def compute_portfolio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio-level metrics (profit/loss, ROI, cumulative rentability).
        
        Assumes df has a 'Ticker' column set to 'Portfolio' and aggregate values
        for Investment ($), Current Value, and Rentability.
        
        Args:
            df: DataFrame with aggregate portfolio data
            
        Returns:
            DataFrame with calculated Profit/Loss, ROI (%), and Cumulative Rentability
        """
        df = df.copy()
        
        # Use dedicated column functions for consistency
        df = (df.pipe(self.add_profit_loss_column)
                .pipe(self.add_roi_column)
                .pipe(self.add_cumulative_rentability_column))
        
        return df
    
    # -------------------
    # Column Addition Helpers
    # -------------------
    
    def add_profit_loss_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Profit/Loss column to DataFrame.
        
        Args:
            df: DataFrame with 'Current Value' and 'Investment ($)' columns
            
        Returns:
            DataFrame with added 'Profit/Loss' column
        """
        df = df.copy()
        df['Profit/Loss'] = df['Current Value'] - df['Investment ($)']
        return df
    
    def add_roi_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ROI (%) column to DataFrame using vectorized operations.
        
        Args:
            df: DataFrame with 'Current Value' and 'Investment ($)' columns
            
        Returns:
            DataFrame with added 'ROI (%)' column
        """
        df = df.copy()
        mask = df['Investment ($)'] != 0
        df['ROI (%)'] = 0.0
        df.loc[mask, 'ROI (%)'] = (
            (df.loc[mask, 'Current Value'] - df.loc[mask, 'Investment ($)']) / 
            df.loc[mask, 'Investment ($)']
        ) * 100
        return df
    
    def add_cumulative_rentability_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Cumulative Rentability column to DataFrame.
        
        Args:
            df: DataFrame with 'Rentability' column
            
        Returns:
            DataFrame with added 'Cumulative Rentability' column
        """
        df = df.copy()
        # Handle NaN values in rentability
        df['Rentability'] = df['Rentability'].fillna(0)
        df['Cumulative Rentability'] = ((1 + df['Rentability']).cumprod() - 1) * 100
        return df
