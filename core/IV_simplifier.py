import pandas as pd
from typing import List, Optional

class IV_simplifier:
    """
    Transforms comprehensive financial data into a simplified version 
    for intrinsic value calculations.
    
    Responsibilities:
    - Transform column names to Year-based format
    - Reorganize rows with Date at the top
    - Filter to essential financial metrics for valuation
    - Remove columns with missing data
    """
    
    def __init__(self, comprehensive_df: pd.DataFrame):
        """
        Initialize with the comprehensive DataFrame.
        
        Args:
            comprehensive_df: DataFrame containing comprehensive financial data
        """
        self.comprehensive_df = comprehensive_df.copy()
        self.simplified_df = None
    
    def _transform_columns_to_years(self) -> 'IV_simplifier':
        """
        Transform column names to Year-based format (Year 1, Year 2, etc.)
        and add Date row with actual years.
        """
        # Extract years from column names and add as Date row
        years = [pd.to_datetime(col).year for col in self.comprehensive_df.columns]
        self.comprehensive_df.loc["Date"] = years
        
        # Create new column names (Year 1, Year 2, etc.)
        num_cols = len(self.comprehensive_df.columns)
        new_col_names = [f"Year {num_cols - i}" for i in range(num_cols)]
        self.comprehensive_df.columns = new_col_names
        
        # Reorder rows to put Date at the top
        self.comprehensive_df = self.comprehensive_df.loc[
            ["Date"] + [idx for idx in self.comprehensive_df.index if idx != "Date"]
        ]
        
        return self
    
    def _filter_essential_metrics(self) -> 'IV_simplifier':
        """
        Filter DataFrame to keep only essential metrics for intrinsic value calculation.
        """
        rows_to_keep = [
            'Date',
            'Ticker',
            'P/E Ratio',
            'Basic EPS',
            'Free Cash Flow',
            'Operating Cash Flow',
            'Investing Cash Flow',   # optional, used for capex adjustments in DCF
            'Dividends',
            'Net Income',
            'Diluted EPS',
            'Share Price',           # optional for P/E
            'Total Assets',
            'Total Liabilities Net Minority Interest',
            'Total Equity Gross Minority Interest',
            'Basic Average Shares',  # optional to convert totals to per-share
            'Diluted Average Shares' # optional to convert totals to per-share
        ]
        
        # Keep only rows that exist in the DataFrame and are in our essential list
        existing_rows = [row for row in rows_to_keep if row in self.comprehensive_df.index]
        self.simplified_df = self.comprehensive_df.loc[existing_rows]
        
        return self
    
    def _remove_empty_columns(self) -> 'IV_simplifier':
        """
        Remove columns with any missing data to ensure data quality.
        """
        if self.simplified_df is not None:
            self.simplified_df = self.simplified_df.dropna(axis=1, how='any')
        return self
    
    def simplify(self) -> pd.DataFrame:
        """
        Execute the complete simplification pipeline.
        
        Returns:
            pd.DataFrame: Simplified DataFrame ready for intrinsic value analysis
        """
        return (self._transform_columns_to_years()
                ._filter_essential_metrics()
                ._remove_empty_columns()
                .get_simplified_data())
    
    def get_simplified_data(self) -> pd.DataFrame:
        """
        Get the final simplified DataFrame.
        
        Returns:
            pd.DataFrame: Simplified financial data for valuation
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