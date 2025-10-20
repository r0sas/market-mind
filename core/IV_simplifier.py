import pandas as pd
from typing import List, Optional


class IVSimplifier:
    """
    Transforms comprehensive financial data into a simplified version 
    for intrinsic value calculations.
    
    Responsibilities:
    - Transform column names to Year-based format
    - Reorganize rows with Date at the top
    - Filter to essential financial metrics for valuation
    - Remove columns with missing data
    """
    
    # Essential metrics for intrinsic value calculation
    ESSENTIAL_METRICS = [
        'Date',
        'Ticker',
        'P/E Ratio',
        'Basic EPS',
        'Free Cash Flow',
        'Operating Cash Flow',
        'Investing Cash Flow',
        'Annual Dividends',
        'Net Income',
        'Diluted EPS',
        'Share Price',
        'Total Assets',
        'Total Liabilities Net Minority Interest',
        'Total Equity Gross Minority Interest',
        'Basic Average Shares',
        'Diluted Average Shares'
    ]
    
    def __init__(self, comprehensive_df: pd.DataFrame):
        """
        Initialize with the comprehensive DataFrame.
        
        Args:
            comprehensive_df: DataFrame containing comprehensive financial data
            
        Raises:
            ValueError: If comprehensive_df is empty or None
        """
        if comprehensive_df is None or comprehensive_df.empty:
            raise ValueError("comprehensive_df cannot be None or empty")
        
        self.comprehensive_df = comprehensive_df.copy()
        self.simplified_df: Optional[pd.DataFrame] = None
    
    def _transform_columns_to_years(self) -> 'IVSimplifier':
        """
        Transform column names to Year-based format (Year 1, Year 2, etc.)
        and add Date row with actual years.
        
        Returns:
            Self for method chaining
        """
        try:
            # Extract years from column names and add as Date row
            years = [pd.to_datetime(col).year for col in self.comprehensive_df.columns]
            self.comprehensive_df.loc["Date"] = years
            
            # Create new column names (Year 1, Year 2, etc.)
            # Year 1 is most recent, increasing numbers go back in time
            num_cols = len(self.comprehensive_df.columns)
            new_col_names = [f"Year {num_cols - i}" for i in range(num_cols)]
            self.comprehensive_df.columns = new_col_names
            
            # Reorder rows to put Date at the top
            other_rows = [idx for idx in self.comprehensive_df.index if idx != "Date"]
            self.comprehensive_df = self.comprehensive_df.loc[["Date"] + other_rows]
            
        except Exception as e:
            raise ValueError(f"Failed to transform columns to years: {e}")
        
        return self
    
    def _filter_essential_metrics(self) -> 'IVSimplifier':
        """
        Filter DataFrame to keep only essential metrics for intrinsic value calculation.
        
        Returns:
            Self for method chaining
        """
        # Keep only rows that exist in the DataFrame and are in our essential list
        existing_rows = [
            row for row in self.ESSENTIAL_METRICS 
            if row in self.comprehensive_df.index
        ]
        
        if not existing_rows:
            raise ValueError("No essential metrics found in the DataFrame")
        
        self.simplified_df = self.comprehensive_df.loc[existing_rows]
        
        return self
    
    def _remove_empty_columns(self) -> 'IVSimplifier':
        """
        Remove columns with any missing data to ensure data quality.
        
        Returns:
            Self for method chaining
        """
        if self.simplified_df is None:
            raise ValueError("simplified_df is None. Call _filter_essential_metrics first")
        
        initial_cols = len(self.simplified_df.columns)
        self.simplified_df = self.simplified_df.dropna(axis=1, how='any')
        final_cols = len(self.simplified_df.columns)
        
        if final_cols == 0:
            raise ValueError("All columns removed due to missing data")
        
        if final_cols < initial_cols:
            print(f"Info: Removed {initial_cols - final_cols} column(s) with missing data")
        
        return self
    
    def simplify(self) -> pd.DataFrame:
        """
        Execute the complete simplification pipeline.
        
        Returns:
            pd.DataFrame: Simplified DataFrame ready for intrinsic value analysis
            
        Raises:
            ValueError: If simplification fails at any step
        """
        try:
            (self._transform_columns_to_years()
                 ._filter_essential_metrics()
                 ._remove_empty_columns())
            
            return self.get_simplified_data()
            
        except Exception as e:
            raise ValueError(f"Simplification failed: {e}")
    
    def get_simplified_data(self) -> pd.DataFrame:
        """
        Get the final simplified DataFrame.
        
        Returns:
            pd.DataFrame: Simplified financial data for valuation
            
        Raises:
            ValueError: If simplified data is not available
        """
        if self.simplified_df is None:
            raise ValueError("No simplified data available. Call simplify() first.")
        return self.simplified_df.copy()
    
    def get_original_data(self) -> pd.DataFrame:
        """
        Get the original comprehensive DataFrame (with column transformations applied).
        
        Returns:
            pd.DataFrame: Original data with year-based columns
        """
        return self.comprehensive_df.copy()
    
    def display_summary(self) -> None:
        """Display a summary of the simplified data"""
        if self.simplified_df is None:
            print("No simplified data available. Call simplify() first.")
            return
        
        print(f"\n{'='*50}")
        print("SIMPLIFIED DATA SUMMARY")
        print(f"{'='*50}")
        print(f"Metrics included: {len(self.simplified_df)}")
        print(f"Years of data: {len(self.simplified_df.columns)}")
        
        if 'Date' in self.simplified_df.index:
            years = self.simplified_df.loc['Date'].values
            print(f"Year range: {int(min(years))} - {int(max(years))}")
        
        if 'Ticker' in self.simplified_df.index:
            ticker = self.simplified_df.loc['Ticker'].iloc[0]
            print(f"Ticker: {ticker}")
        
        print(f"\nMetrics available:")
        for metric in self.simplified_df.index:
            if metric not in ['Date', 'Ticker']:
                print(f"  - {metric}")