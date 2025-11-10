import pandas as pd
from typing import Optional, Dict, Any, List
import logging
from core.datafetcher.data_fetcher import DataFetcher
from core.datafetcher.constants import EPS_COLUMNS, BALANCE_SHEET_ITEMS, CASH_FLOW_ITEMS
from core.datafetcher.metrics_calculator import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataProcessorError(Exception):
    """Custom exception for FinancialDataProcessor errors"""
    pass


class FinancialDataProcessor(DataFetcher):
    """Class to process and enhance financial data fetched using DataFetcher."""
    
    def __init__(self, ticker_symbol: str):
        super().__init__(ticker_symbol)
    
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
    
    def get_comprehensive_data(self) -> pd.DataFrame:
        """
        Fetch and combine income statement, balance sheet, and cash flow data.
        
        Returns:
            Combined DataFrame with all financial data
        """
        try:
            logger.info(f"Fetching data for {self.ticker_symbol}...")
            income_stmt = self.get_income_statement()
            balance_sheet = self.get_balance_sheet()
            cash_flow = self.get_cash_flow()
            share_price_data = self.get_share_price_data(period="max")  # Use max for historical data
            dividends = self.get_dividends()
            summary = self.get_summary()

            if income_stmt.empty:
                raise FinancialDataProcessorError(
                    f"No income statement data to process for ticker: {self.ticker_symbol}"
                )
            
            logger.info(f"Creating enhanced dataset with {len(income_stmt.columns)} periods...")
            enhanced_income_stmt = self._create_enhanced_income_statement(
                income_stmt, 
                share_price_data, 
                dividends
            )
            combined_df = self._create_combined_dataframe(
                enhanced_income_stmt, 
                balance_sheet, 
                cash_flow, 
                summary
            )
            logger.info(f"Successfully created comprehensive dataset: {combined_df.shape}")
            return combined_df
            
        except FinancialDataProcessorError:
            raise
        except Exception as e:
            raise FinancialDataProcessorError(
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
        enhanced_income_stmt = income_stmt.copy()
        
        # Add ticker symbol
        enhanced_income_stmt.loc['Ticker'] = [self.ticker_symbol] * len(enhanced_income_stmt.columns)
        
        # Add share prices
        share_prices = self._get_share_prices_for_dates(
            enhanced_income_stmt.columns, 
            share_price_data
        )
        enhanced_income_stmt.loc['Share Price'] = share_prices
        
        # Add total annual dividends
        dividend_values = self._get_annual_dividends_for_dates(
            enhanced_income_stmt.columns, 
            dividends
        )
        enhanced_income_stmt.loc['Annual Dividends'] = dividend_values

        # Calculate P/E ratios using MetricsCalculator
        pe_ratios = MetricsCalculator.calculate_pe_ratios(
            enhanced_income_stmt,
            EPS_COLUMNS
        )
        enhanced_income_stmt.loc["P/E Ratio"] = pe_ratios

        return enhanced_income_stmt
    
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
            year_start = fiscal_date - pd.DateOffset(years=1)
            mask = (dividends_df['Date'] > year_start) & (dividends_df['Date'] <= fiscal_date)
            annual_div = dividends_df[mask]['Dividends'].sum() if mask.any() else 0.0
            dividend_values.append(annual_div)
        
        return dividend_values
    
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


# Example usage
if __name__ == "__main__":
    from core.datafetcher.data_displayer import DataDisplayer
    
    try:
        # Create processor instance
        processor = FinancialDataProcessor("MSFT")
        
        # Get all data in one comprehensive DataFrame
        comprehensive_df = processor.get_comprehensive_data()
        
        # Display overview
        summary = processor.get_summary()
        DataDisplayer.display_data_overview(
            processor.ticker_symbol,
            summary,
            comprehensive_df
        )
        
        # Show sample of combined data
        print("\nFirst few rows of combined data:")
        print(comprehensive_df.head(10))
        
    except FinancialDataProcessorError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")