import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import date
import logging
from test.Config import (
    EPS_COLUMNS, 
    BALANCE_SHEET_ITEMS, 
    CASH_FLOW_ITEMS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcherError(Exception):
    """Custom exception for DataFetcher errors"""
    pass


class DataFetcher:
    """
    Fetch and process financial data from Yahoo Finance.
    
    This class handles data retrieval, validation, and preprocessing
    for financial statements and market data.
    
    Example:
        >>> fetcher = DataFetcher("AAPL")
        >>> data = fetcher.get_comprehensive_data()
        >>> print(data.shape)
    """
    
    def __init__(self, ticker_symbol: str):
        """
        Initialize DataFetcher with a ticker symbol.
        
        Args:
            ticker_symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
            
        Raises:
            ValueError: If ticker_symbol is empty or invalid
        """
        if not ticker_symbol or not isinstance(ticker_symbol, str):
            raise ValueError("ticker_symbol must be a non-empty string")
        
        self.ticker_symbol = ticker_symbol.upper().strip()
        self.ticker = yf.Ticker(ticker_symbol)
        self._cache = {}  # Simple in-memory cache
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get basic company information.
        
        Returns:
            Dictionary containing company information
            
        Raises:
            DataFetcherError: If info cannot be fetched
        """
        if 'info' in self._cache:
            return self._cache['info']
        
        try:
            info = self.ticker.info
            if not info or 'symbol' not in info:
                raise DataFetcherError(
                    f"Invalid or missing data for {self.ticker_symbol}. "
                    "Please verify the ticker symbol is correct."
                )
            self._cache['info'] = info
            return info
        except Exception as e:
            raise DataFetcherError(
                f"Failed to fetch info for {self.ticker_symbol}: {str(e)}"
            )
        
    def get_income_statement(self) -> pd.DataFrame:
        """
        Get income statement data.
        
        Returns:
            DataFrame with income statement data
            
        Raises:
            DataFetcherError: If income statement cannot be fetched
        """
        if 'income_stmt' in self._cache:
            return self._cache['income_stmt']
        
        try:
            stmt = self.ticker.income_stmt
            if stmt is None or stmt.empty:
                raise DataFetcherError(
                    f"No income statement data available for {self.ticker_symbol}"
                )
            self._cache['income_stmt'] = stmt
            return stmt
        except Exception as e:
            raise DataFetcherError(
                f"Failed to fetch income statement for {self.ticker_symbol}: {str(e)}"
            )

    def get_balance_sheet(self) -> pd.DataFrame:
        """
        Get balance sheet data.
        
        Returns:
            DataFrame with balance sheet data
            
        Raises:
            DataFetcherError: If balance sheet cannot be fetched
        """
        if 'balance_sheet' in self._cache:
            return self._cache['balance_sheet']
        
        try:
            bs = self.ticker.balance_sheet
            if bs is None or bs.empty:
                raise DataFetcherError(
                    f"No balance sheet data available for {self.ticker_symbol}"
                )
            self._cache['balance_sheet'] = bs
            return bs
        except Exception as e:
            raise DataFetcherError(
                f"Failed to fetch balance sheet for {self.ticker_symbol}: {str(e)}"
            )

    def get_cash_flow(self) -> pd.DataFrame:
        """
        Get cash flow statement data.
        
        Returns:
            DataFrame with cash flow data
            
        Raises:
            DataFetcherError: If cash flow cannot be fetched
        """
        if 'cashflow' in self._cache:
            return self._cache['cashflow']
        
        try:
            cf = self.ticker.cashflow
            if cf is None or cf.empty:
                raise DataFetcherError(
                    f"No cash flow data available for {self.ticker_symbol}"
                )
            self._cache['cashflow'] = cf
            return cf
        except Exception as e:
            raise DataFetcherError(
                f"Failed to fetch cash flow for {self.ticker_symbol}: {str(e)}"
            )

    def get_share_price_data(self, period: str = "max") -> pd.DataFrame:
        """
        Get historical share price data.
        
        Args:
            period: Time period (valid: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataFetcherError: If share price data cannot be fetched
        """
        cache_key = f'share_price_{period}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            share_price_data = self.ticker.history(period=period)
            if share_price_data.empty:
                raise DataFetcherError(
                    f"No share price data available for {self.ticker_symbol}"
                )
            share_price_data.reset_index(inplace=True)
            self._cache[cache_key] = share_price_data
            return share_price_data
        except Exception as e:
            raise DataFetcherError(
                f"Failed to fetch share price data for {self.ticker_symbol}: {str(e)}"
            )
    
    def get_dividends(self) -> pd.Series:
        """
        Get dividend data.
        
        Returns:
            Series with dividend data (may be empty if no dividends)
            
        Raises:
            DataFetcherError: If dividend data cannot be fetched
        """
        if 'dividends' in self._cache:
            return self._cache['dividends']
        
        try:
            divs = self.ticker.dividends
            # Dividends can be empty for non-dividend-paying stocks
            self._cache['dividends'] = divs
            return divs
        except Exception as e:
            logger.warning(f"Failed to fetch dividends for {self.ticker_symbol}: {str(e)}")
            return pd.Series()  # Return empty series instead of raising
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key metrics.
        
        Returns:
            Dictionary with key financial metrics
        """
        info = self.get_info()
        summary = {
            'company_name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap'),
            'current_price': info.get('currentPrice'),
            'pe_ratio': info.get('trailingPE'),
            'eps': info.get('trailingEps'),
            'dividend_yield': info.get('dividendYield')
        }
        return summary
    
    def calculate_pe_ratio(
        self, 
        price: Optional[float], 
        earnings_per_share: Optional[float]
    ) -> Optional[float]:
        """
        Calculate P/E ratio.
        
        Args:
            price: Share price
            earnings_per_share: EPS value
            
        Returns:
            P/E ratio or None if calculation not possible
        """
        if not price or not earnings_per_share or earnings_per_share == 0:
            return None
        return price / earnings_per_share

    def get_comprehensive_data(self) -> pd.DataFrame:
        """
        Get all financial data and combine into a comprehensive DataFrame.
        
        This method fetches and combines:
        - Income statement
        - Balance sheet  
        - Cash flow statement
        - Share price history
        - Dividend data
        - Company summary
        
        Returns:
            Combined DataFrame with all financial data
            
        Raises:
            DataFetcherError: If critical data cannot be fetched or combined
        """
        try:
            # Get financial statements
            logger.info(f"Fetching data for {self.ticker_symbol}...")
            income_stmt = self.get_income_statement()
            balance_sheet = self.get_balance_sheet()
            cash_flow = self.get_cash_flow()
            
            # Get market data
            share_price_data = self.get_share_price_data()
            dividends = self.get_dividends()
            summary = self.get_summary()
            
            # Validate minimum data requirements
            if income_stmt.empty:
                raise DataFetcherError("Income statement is empty")
            
            logger.info(f"Creating enhanced dataset with {len(income_stmt.columns)} periods...")
            
            # Create enhanced income statement with additional metrics
            enhanced_income_stmt = self._create_enhanced_income_statement(
                income_stmt, share_price_data, dividends
            )
            
            # Combine everything into a master DataFrame
            combined_df = self._create_combined_dataframe(
                enhanced_income_stmt, balance_sheet, cash_flow, summary
            )
            
            logger.info(f"Successfully created comprehensive dataset: {combined_df.shape}")
            return combined_df
            
        except DataFetcherError:
            raise
        except Exception as e:
            raise DataFetcherError(
                f"Failed to create comprehensive data for {self.ticker_symbol}: {str(e)}"
            )
    
    def _create_enhanced_income_statement(
        self, 
        income_stmt: pd.DataFrame, 
        share_price_data: pd.DataFrame, 
        dividends: pd.Series
    ) -> pd.DataFrame:
        """
        Create enhanced income statement with share price, dividends, and P/E ratio.
        
        Args:
            income_stmt: Income statement DataFrame
            share_price_data: Historical price data
            dividends: Dividend data
            
        Returns:
            Enhanced income statement DataFrame
        """
        enhanced_is = income_stmt.copy()
        
        # Add ticker symbol
        enhanced_is.loc['Ticker'] = [self.ticker_symbol] * len(enhanced_is.columns)
        
        # Add share prices
        share_prices = self._get_share_prices_for_dates(enhanced_is.columns, share_price_data)
        enhanced_is.loc['Share Price'] = share_prices
        
        # Add total annual dividends
        dividend_values = self._get_annual_dividends_for_dates(enhanced_is.columns, dividends)
        enhanced_is.loc['Annual Dividends'] = dividend_values
        
        # Add P/E ratios
        pe_ratios = self._calculate_pe_ratios(enhanced_is)
        enhanced_is.loc["P/E Ratio"] = pe_ratios
        
        return enhanced_is
    
    def _get_share_prices_for_dates(
        self, 
        fiscal_dates: pd.Index, 
        share_price_data: pd.DataFrame
    ) -> List[Optional[float]]:
        """
        Get share prices for each fiscal date.
        
        Args:
            fiscal_dates: Index of fiscal period end dates
            share_price_data: DataFrame with historical prices
            
        Returns:
            List of share prices matching fiscal dates
        """
        share_prices = []
        for date_str in fiscal_dates:
            fiscal_date = pd.to_datetime(date_str).date()
            mask = share_price_data['Date'].dt.date <= fiscal_date
            if mask.any():
                last_row = share_price_data[mask].iloc[-1]
                share_prices.append(last_row['Close'])
            else:
                logger.warning(f"No share price data available for date {fiscal_date}")
                share_prices.append(None)
        return share_prices
    
    def _get_annual_dividends_for_dates(
        self, 
        fiscal_dates: pd.Index, 
        dividends: pd.Series
    ) -> List[float]:
        """
        Calculate total annual dividends for each fiscal year.
        
        Args:
            fiscal_dates: Index of fiscal period end dates
            dividends: Series with dividend data
            
        Returns:
            List of annual dividend totals
        """
        dividend_values = []
        
        if dividends is None or dividends.empty:
            return [0.0] * len(fiscal_dates)
        
        try:
            dividends_df = dividends.reset_index()
            dividends_df['Date'] = pd.to_datetime(dividends_df['Date']).dt.tz_localize(None)
        except Exception as e:
            logger.warning(f"Error processing dividend dates: {str(e)}")
            return [0.0] * len(fiscal_dates)
        
        for date_str in fiscal_dates:
            fiscal_date = pd.to_datetime(date_str).tz_localize(None)
            # Sum dividends in the year prior to fiscal date
            year_start = fiscal_date - pd.DateOffset(years=1)
            mask = (dividends_df['Date'] > year_start) & (dividends_df['Date'] <= fiscal_date)
            annual_div = dividends_df[mask]['Dividends'].sum() if mask.any() else 0.0
            dividend_values.append(annual_div)
        
        return dividend_values
    
    def _calculate_pe_ratios(self, enhanced_is: pd.DataFrame) -> List[Optional[float]]:
        """
        Calculate P/E ratios for each period.
        
        Args:
            enhanced_is: Enhanced income statement with share prices
            
        Returns:
            List of P/E ratios
        """
        pe_ratios = []
        for col in enhanced_is.columns:
            share_price = enhanced_is.loc["Share Price", col] if "Share Price" in enhanced_is.index else None
            
            # Try to find EPS from available columns
            eps = None
            for eps_name in EPS_COLUMNS:
                if eps_name in enhanced_is.index:
                    eps = enhanced_is.loc[eps_name, col]
                    break
            
            pe = self.calculate_pe_ratio(share_price, eps)
            pe_ratios.append(pe)
        
        return pe_ratios
    
    def _create_combined_dataframe(
        self, 
        enhanced_income_stmt: pd.DataFrame, 
        balance_sheet: pd.DataFrame, 
        cash_flow: pd.DataFrame, 
        summary: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Combine all data into a single DataFrame.
        
        Args:
            enhanced_income_stmt: Enhanced income statement
            balance_sheet: Balance sheet data
            cash_flow: Cash flow data
            summary: Company summary metrics
            
        Returns:
            Combined DataFrame with all data
        """
        combined_df = enhanced_income_stmt.copy()
        
        # Add key balance sheet items
        for item in BALANCE_SHEET_ITEMS:
            if item in balance_sheet.index:
                combined_df.loc[item] = balance_sheet.loc[item]
            else:
                logger.debug(f"Balance sheet item '{item}' not found")
        
        # Add key cash flow items
        for item in CASH_FLOW_ITEMS:
            if item in cash_flow.index:
                combined_df.loc[item] = cash_flow.loc[item]
            else:
                logger.debug(f"Cash flow item '{item}' not found")
        
        # Add summary metrics as new rows (only if values exist)
        num_cols = len(combined_df.columns)
        if summary['market_cap']:
            combined_df.loc['Market Cap'] = [summary['market_cap']] * num_cols
        if summary['current_price']:
            combined_df.loc['Current Price'] = [summary['current_price']] * num_cols
        if summary['sector']:
            combined_df.loc['Sector'] = [summary['sector']] * num_cols
        if summary['industry']:
            combined_df.loc['Industry'] = [summary['industry']] * num_cols
        
        return combined_df
    
    def display_data_overview(self, combined_df: pd.DataFrame) -> None:
        """
        Display an overview of fetched data.
        
        Args:
            combined_df: The combined DataFrame to summarize
        """
        summary = self.get_summary()
        
        print(f"\n{'='*50}")
        print(f"DATA OVERVIEW FOR {self.ticker_symbol}")
        print(f"{'='*50}")
        
        print(f"Company: {summary['company_name']}")
        print(f"Sector: {summary['sector']}")
        print(f"Industry: {summary['industry']}")
        
        if summary['market_cap']:
            print(f"Market Cap: ${summary['market_cap']:,.0f}")
        if summary['current_price']:
            print(f"Current Price: ${summary['current_price']:.2f}")
        if summary['pe_ratio']:
            print(f"P/E Ratio: {summary['pe_ratio']:.2f}")
        
        print(f"\nCombined DataFrame shape: {combined_df.shape}")
        print(f"Time periods covered: {len(combined_df.columns)}")


# Example usage
if __name__ == "__main__":
    try:
        # Create fetcher instance
        fetcher = DataFetcher("MSFT")
        
        # Get all data in one comprehensive DataFrame
        comprehensive_df = fetcher.get_comprehensive_data()
        
        # Display overview
        fetcher.display_data_overview(comprehensive_df)
        
        # Show sample of combined data
        print("\nFirst few rows of combined data:")
        print(comprehensive_df.head(10))
        
    except DataFetcherError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")