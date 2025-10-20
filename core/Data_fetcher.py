import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import date

class DataFetcher:
    """Fetch and process financial data from Yahoo Finance"""
    
    # Constants for column names
    EPS_COLUMNS = ["Basic EPS", "Diluted EPS", "Earnings Per Share"]
    BALANCE_SHEET_ITEMS = [
        'Total Assets', 
        'Total Liabilities Net Minority Interest', 
        'Total Equity Gross Minority Interest', 
        'Current Assets', 
        'Current Liabilities'
    ]
    CASH_FLOW_ITEMS = [
        'Operating Cash Flow', 
        'Investing Cash Flow', 
        'Financing Cash Flow', 
        'Free Cash Flow'
    ]
    
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(ticker_symbol)
    
    def get_info(self) -> Dict[str, Any]:
        """Get basic company information"""
        try:
            return self.ticker.info
        except Exception as e:
            raise ValueError(f"Failed to fetch info for {self.ticker_symbol}: {e}")
        
    def get_income_statement(self) -> pd.DataFrame:
        """Get income statement data"""
        try:
            return self.ticker.income_stmt
        except Exception as e:
            raise ValueError(f"Failed to fetch income statement for {self.ticker_symbol}: {e}")

    def get_balance_sheet(self) -> pd.DataFrame:
        """Get balance sheet data"""
        try:
            return self.ticker.balance_sheet
        except Exception as e:
            raise ValueError(f"Failed to fetch balance sheet for {self.ticker_symbol}: {e}")

    def get_cash_flow(self) -> pd.DataFrame:
        """Get cash flow statement data"""
        try:
            return self.ticker.cashflow
        except Exception as e:
            raise ValueError(f"Failed to fetch cash flow for {self.ticker_symbol}: {e}")

    def get_share_price_data(self, period: str = "max") -> pd.DataFrame:
        """Get historical share price data"""
        try:
            share_price_data = self.ticker.history(period=period)
            if share_price_data.empty:
                raise ValueError(f"No share price data available for {self.ticker_symbol}")
            share_price_data.reset_index(inplace=True)
            return share_price_data
        except Exception as e:
            raise ValueError(f"Failed to fetch share price data for {self.ticker_symbol}: {e}")
    
    def get_dividends(self) -> pd.Series:
        """Get dividend data"""
        try:
            return self.ticker.dividends
        except Exception as e:
            raise ValueError(f"Failed to fetch dividends for {self.ticker_symbol}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics"""
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
    
    def calculate_pe_ratio(self, price: Optional[float], earnings_per_share: Optional[float]) -> Optional[float]:
        """Calculate P/E ratio"""
        if not price or not earnings_per_share or earnings_per_share == 0:
            return None
        return price / earnings_per_share

    def get_comprehensive_data(self) -> pd.DataFrame:
        """
        Get all financial data and combine into a comprehensive DataFrame
        
        Returns:
            pandas.DataFrame: Combined data from all sources
            
        Raises:
            ValueError: If critical data cannot be fetched
        """
        # Get financial statements
        income_stmt = self.get_income_statement()
        balance_sheet = self.get_balance_sheet()
        cash_flow = self.get_cash_flow()
        
        # Get market data
        share_price_data = self.get_share_price_data()
        dividends = self.get_dividends()
        summary = self.get_summary()
        
        # Create enhanced income statement with additional metrics
        enhanced_income_stmt = self._create_enhanced_income_statement(
            income_stmt, share_price_data, dividends
        )
        
        # Combine everything into a master DataFrame
        combined_df = self._create_combined_dataframe(
            enhanced_income_stmt, balance_sheet, cash_flow, summary
        )
        
        return combined_df
    
    def _create_enhanced_income_statement(
        self, 
        income_stmt: pd.DataFrame, 
        share_price_data: pd.DataFrame, 
        dividends: pd.Series
    ) -> pd.DataFrame:
        """Create enhanced income statement with share price, dividends, and P/E ratio"""
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
        """Get share prices for each fiscal date"""
        share_prices = []
        for date_str in fiscal_dates:
            fiscal_date = pd.to_datetime(date_str).date()
            mask = share_price_data['Date'].dt.date <= fiscal_date
            if mask.any():
                last_row = share_price_data[mask].iloc[-1]
                share_prices.append(last_row['Close'])
            else:
                share_prices.append(None)
        return share_prices
    
    def _get_annual_dividends_for_dates(
        self, 
        fiscal_dates: pd.Index, 
        dividends: pd.Series
    ) -> List[float]:
        """Calculate total annual dividends for each fiscal year"""
        dividend_values = []
        
        if dividends is None or dividends.empty:
            return [0.0] * len(fiscal_dates)
        
        dividends_df = dividends.reset_index()
        dividends_df['Date'] = pd.to_datetime(dividends_df['Date']).dt.tz_localize(None)
        
        for date_str in fiscal_dates:
            fiscal_date = pd.to_datetime(date_str).tz_localize(None)
            # Sum dividends in the year prior to fiscal date
            year_start = fiscal_date - pd.DateOffset(years=1)
            mask = (dividends_df['Date'] > year_start) & (dividends_df['Date'] <= fiscal_date)
            annual_div = dividends_df[mask]['Dividends'].sum() if mask.any() else 0.0
            dividend_values.append(annual_div)
        
        return dividend_values
    
    def _calculate_pe_ratios(self, enhanced_is: pd.DataFrame) -> List[Optional[float]]:
        """Calculate P/E ratios for each period"""
        pe_ratios = []
        for col in enhanced_is.columns:
            share_price = enhanced_is.loc["Share Price", col]
            
            # Try to find EPS from available columns
            eps = None
            for eps_name in self.EPS_COLUMNS:
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
        """Combine all data into a single DataFrame"""
        combined_df = enhanced_income_stmt.copy()
        
        # Add key balance sheet items
        for item in self.BALANCE_SHEET_ITEMS:
            if item in balance_sheet.index:
                combined_df.loc[item] = balance_sheet.loc[item]
        
        # Add key cash flow items
        for item in self.CASH_FLOW_ITEMS:
            if item in cash_flow.index:
                combined_df.loc[item] = cash_flow.loc[item]
        
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
        """Display an overview of fetched data"""
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
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")