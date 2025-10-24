import pandas as pd
from typing import List, Optional, Tuple
import logging
from core.Config import ESSENTIAL_METRICS, MIN_HISTORICAL_YEARS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifierError(Exception):
    """Custom exception for IVSimplifier errors"""
    pass


class IVSimplifier:
    """
    Transforms comprehensive financial data into a simplified version 
    for intrinsic value calculations.
    
    Responsibilities:
    - Transform column names to Year-based format
    - Reorganize rows with Date at the top
    - Filter to essential financial metrics for valuation
    - Handle missing data intelligently
    
    Example:
        >>> simplifier = IVSimplifier(comprehensive_df)
        >>> simplified = simplifier.simplify()
        >>> print(simplified.shape)
    """
    
    def __init__(
        self, 
        comprehensive_df: pd.DataFrame,
        essential_metrics: Optional[List[str]] = None,
        prioritize_recent: bool = True
    ):
        """
        Initialize with the comprehensive DataFrame.
        
        Args:
            comprehensive_df: DataFrame containing comprehensive financial data
            essential_metrics: List of metrics to keep (default: from config)
            prioritize_recent: If True, prefer keeping recent years over complete history
            
        Raises:
            SimplifierError: If comprehensive_df is empty or None
        """
        if comprehensive_df is None or comprehensive_df.empty:
            raise SimplifierError("comprehensive_df cannot be None or empty")
        
        self.comprehensive_df = comprehensive_df.copy()
        self.simplified_df: Optional[pd.DataFrame] = None
        self.essential_metrics = essential_metrics or ESSENTIAL_METRICS
        self.prioritize_recent = prioritize_recent
        self.removed_columns: List[str] = []
        self.missing_metrics: List[str] = []
        self.data_quality_warnings: List[str] = []
    
    def _transform_columns_to_years(self) -> 'IVSimplifier':
        """
        Transform column names to Year-based format (Year 1, Year 2, etc.)
        and add Date row with actual years.
        
        Returns:
            Self for method chaining
            
        Raises:
            SimplifierError: If date transformation fails
        """
        try:
            # Extract years from column names and add as Date row
            years = []
            for col in self.comprehensive_df.columns:
                try:
                    year = pd.to_datetime(col).year
                    years.append(year)
                except Exception as e:
                    logger.warning(f"Could not parse date from column '{col}': {e}")
                    years.append(None)
            
            if all(y is None for y in years):
                raise SimplifierError("Could not parse any valid dates from column names")
            
            self.comprehensive_df.loc["Date"] = years
            
            # Create new column names (Year 1 is most recent, increasing numbers go back in time)
            num_cols = len(self.comprehensive_df.columns)
            new_col_names = [f"Year {num_cols - i}" for i in range(num_cols)]
            self.comprehensive_df.columns = new_col_names
            
            # Reorder rows to put Date at the top
            other_rows = [idx for idx in self.comprehensive_df.index if idx != "Date"]
            self.comprehensive_df = self.comprehensive_df.loc[["Date"] + other_rows]
            
            logger.info(f"Transformed {num_cols} columns to year-based format")
            
        except SimplifierError:
            raise
        except Exception as e:
            raise SimplifierError(f"Failed to transform columns to years: {str(e)}")
        
        return self
    
    def _filter_essential_metrics(self) -> 'IVSimplifier':
        """
        Filter DataFrame to keep only essential metrics for intrinsic value calculation.
        
        Returns:
            Self for method chaining
            
        Raises:
            SimplifierError: If no essential metrics found
        """
        # Keep only rows that exist in the DataFrame and are in our essential list
        existing_rows = [
            row for row in self.essential_metrics 
            if row in self.comprehensive_df.index
        ]
        
        # Track missing metrics
        self.missing_metrics = [
            row for row in self.essential_metrics
            if row not in self.comprehensive_df.index
        ]
        
        if self.missing_metrics:
            logger.info(f"Missing metrics: {', '.join(self.missing_metrics)}")
        
        if not existing_rows:
            raise SimplifierError(
                "No essential metrics found in the DataFrame. "
                f"Available metrics: {list(self.comprehensive_df.index[:10])}..."
            )
        
        self.simplified_df = self.comprehensive_df.loc[existing_rows]
        logger.info(f"Filtered to {len(existing_rows)} essential metrics")
        
        return self
    
    def _handle_missing_data(self) -> 'IVSimplifier':
        """
        Handle missing data intelligently based on prioritization strategy.
        
        If prioritize_recent is True, keeps most recent complete columns.
        Otherwise, removes any column with missing data.
        
        Returns:
            Self for method chaining
            
        Raises:
            SimplifierError: If all columns would be removed
        """
        if self.simplified_df is None:
            raise SimplifierError("simplified_df is None. Call _filter_essential_metrics first")
        
        initial_cols = len(self.simplified_df.columns)
        
        if self.prioritize_recent:
            # Keep columns from most recent, stopping when we hit missing data
            # in critical metrics
            critical_metrics = ['Free Cash Flow', 'Diluted EPS', 'Basic EPS']
            
            cols_to_keep = []
            for col in self.simplified_df.columns:
                col_data = self.simplified_df[col]
                
                # Check if any critical metric is missing
                has_critical_data = False
                for metric in critical_metrics:
                    if metric in self.simplified_df.index:
                        if pd.notna(col_data[metric]):
                            has_critical_data = True
                            break
                
                # Keep if has critical data or if it's a recent column with mostly complete data
                missing_pct = col_data.isna().sum() / len(col_data)
                if has_critical_data or missing_pct < 0.2:  # Allow 20% missing
                    cols_to_keep.append(col)
            
            if cols_to_keep:
                self.removed_columns = [col for col in self.simplified_df.columns if col not in cols_to_keep]
                self.simplified_df = self.simplified_df[cols_to_keep]
            else:
                # Fallback: just remove columns with any missing data
                self.simplified_df = self.simplified_df.dropna(axis=1, how='any')
                
        else:
            # Original behavior: remove any column with missing data
            self.simplified_df = self.simplified_df.dropna(axis=1, how='any')
        
        final_cols = len(self.simplified_df.columns)
        
        if final_cols == 0:
            raise SimplifierError(
                "All columns removed due to missing data. "
                "The company may not have sufficient financial history."
            )
        
        if final_cols < initial_cols:
            removed_count = initial_cols - final_cols
            logger.info(f"Removed {removed_count} column(s) with missing data")
            
            if final_cols < MIN_HISTORICAL_YEARS:
                logger.warning(
                    f"Only {final_cols} years of data available. "
                    f"Recommend at least {MIN_HISTORICAL_YEARS} years for reliable valuation."
                )
        
        return self
    
    def _validate_data_quality(self) -> Tuple[bool, List[str]]:
        """
        Validate the quality of simplified data and return warnings.
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        if self.simplified_df is None:
            return False, ["Simplified dataframe is None"]
        
        warnings = []
        
        # Check minimum years
        num_years = len(self.simplified_df.columns)
        if num_years < MIN_HISTORICAL_YEARS:
            warnings.append(
                f"Only {num_years} years of data available. "
                f"Recommend at least {MIN_HISTORICAL_YEARS} years."
            )
        
        # Check for critical metrics
        critical_metrics = ['Free Cash Flow', 'Diluted EPS', 'Basic EPS', 'Share Price']
        missing_critical = [m for m in critical_metrics if m not in self.simplified_df.index]
        if missing_critical:
            warnings.append(f"Missing critical metrics: {', '.join(missing_critical)}")
        
        # Check for negative values in key metrics that should be positive
        if 'Total Assets' in self.simplified_df.index:
            if (self.simplified_df.loc['Total Assets'] <= 0).any():
                warnings.append("Total Assets contains non-positive values")
        
        # Check for extreme outliers in P/E ratio
        if 'P/E Ratio' in self.simplified_df.index:
            pe_values = self.simplified_df.loc['P/E Ratio'].dropna()
            if len(pe_values) > 0:
                # Check for negative P/E ratios (negative earnings)
                if (pe_values < 0).any():
                    warnings.append("P/E Ratio contains negative values (company has negative earnings)")
                
                # Check for extremely high P/E ratios (>200 is suspicious)
                if (pe_values > 200).any():
                    max_pe = pe_values.max()
                    warnings.append(f"P/E Ratio contains extremely high values (max: {max_pe:.1f})")
                
                # Check for high volatility in P/E over time
                if len(pe_values) > 1:
                    mean_pe = pe_values.mean()
                    std_pe = pe_values.std()
                    if mean_pe > 0:
                        coefficient_of_variation = std_pe / mean_pe
                        if coefficient_of_variation > 1.0:
                            warnings.append(
                                f"P/E Ratio is highly volatile (CV: {coefficient_of_variation:.2f}). "
                                "P/E model may be unreliable."
                            )
        
        # Check for negative Free Cash Flow
        if 'Free Cash Flow' in self.simplified_df.index:
            fcf_values = self.simplified_df.loc['Free Cash Flow'].dropna()
            if len(fcf_values) > 0:
                negative_fcf_count = (fcf_values < 0).sum()
                if negative_fcf_count > 0:
                    warnings.append(
                        f"Free Cash Flow is negative in {negative_fcf_count} out of {len(fcf_values)} years. "
                        "DCF model may not be reliable."
                    )
        
        # Check if company pays dividends
        if 'Annual Dividends' in self.simplified_df.index:
            div_values = self.simplified_df.loc['Annual Dividends'].dropna()
            if len(div_values) > 0 and div_values.sum() == 0:
                warnings.append("Company does not pay dividends. DDM model will not be applicable.")
        
        # Overall validity check
        is_valid = len(missing_critical) == 0 and num_years >= 2
        
        return is_valid, warnings
    
    def simplify(self) -> pd.DataFrame:
        """
        Execute the complete simplification pipeline.
        
        Returns:
            pd.DataFrame: Simplified DataFrame ready for intrinsic value analysis
            
        Raises:
            SimplifierError: If simplification fails at any step
        """
        try:
            (self._transform_columns_to_years()
                 ._filter_essential_metrics()
                 ._handle_missing_data())
            
            # Validate data quality
            is_valid, warnings = self._validate_data_quality()
            self.data_quality_warnings = warnings
            
            # Log warnings
            if warnings:
                logger.warning("Data quality issues detected:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
            
            if not is_valid:
                logger.error("Data validation failed. Results may be unreliable.")
            
            return self.get_simplified_data()
            
        except Exception as e:
            raise SimplifierError(f"Simplification failed: {str(e)}")
    
    def get_simplified_data(self) -> pd.DataFrame:
        """
        Get the final simplified DataFrame.
        
        Returns:
            pd.DataFrame: Simplified financial data for valuation
            
        Raises:
            SimplifierError: If simplified data is not available
        """
        if self.simplified_df is None:
            raise SimplifierError("No simplified data available. Call simplify() first.")
        return self.simplified_df.copy()
    
    def get_original_data(self) -> pd.DataFrame:
        """
        Get the original comprehensive DataFrame (with column transformations applied).
        
        Returns:
            pd.DataFrame: Original data with year-based columns
        """
        return self.comprehensive_df.copy()
    
    def get_data_quality_report(self) -> Dict[str, any]:
        """
        Get a comprehensive data quality report.
        
        Returns:
            Dictionary with data quality metrics and warnings
        """
        if self.simplified_df is None:
            return {"error": "No simplified data available. Call simplify() first."}
        
        report = {
            "num_years": len(self.simplified_df.columns),
            "num_metrics": len(self.simplified_df.index),
            "missing_metrics": self.missing_metrics,
            "removed_columns": self.removed_columns,
            "warnings": self.data_quality_warnings,
            "has_critical_data": len([m for m in ['Free Cash Flow', 'Diluted EPS', 'Basic EPS'] 
                                     if m in self.simplified_df.index]) >= 2,
        }
        
        # Add year range
        if 'Date' in self.simplified_df.index:
            years = self.simplified_df.loc['Date'].values
            report["year_range"] = f"{int(min(years))} - {int(max(years))}"
        
        return report
    
    def get_model_selection_metrics(self) -> Dict[str, any]:
        """
        Extract metrics specifically for intelligent model selection.
        
        This method provides additional calculated metrics beyond raw data
        to help the ModelSelector determine which valuation models are appropriate.
        
        Returns:
            Dictionary with key metrics for model selection
        """
        if self.simplified_df is None:
            raise SimplifierError("No simplified data available. Call simplify() first.")
        
        metrics = {}
        
        # Dividend metrics
        if 'Annual Dividends' in self.simplified_df.index:
            divs = self.simplified_df.loc['Annual Dividends'].dropna()
            metrics['has_dividends'] = len(divs) > 0 and divs.sum() > 0
            
            if metrics['has_dividends']:
                divs_positive = divs[divs > 0]
                if len(divs_positive) >= 2:
                    # Calculate dividend consistency
                    metrics['dividend_years'] = len(divs_positive)
                    metrics['total_years'] = len(self.simplified_df.columns)
                    
                    # Payout ratio (if EPS available)
                    if 'Diluted EPS' in self.simplified_df.index:
                        eps = self.simplified_df.loc['Diluted EPS'].iloc[0]
                        div = divs.iloc[0]
                        if eps > 0:
                            metrics['payout_ratio'] = div / eps
        else:
            metrics['has_dividends'] = False
        
        # Free Cash Flow metrics
        if 'Free Cash Flow' in self.simplified_df.index:
            fcf = self.simplified_df.loc['Free Cash Flow'].dropna()
            metrics['fcf_years'] = len(fcf)
            metrics['fcf_positive_years'] = (fcf > 0).sum()
            
            if len(fcf) > 0:
                metrics['fcf_consistency'] = (fcf > 0).sum() / len(fcf)
        
        # EPS metrics
        eps_metric = 'Diluted EPS' if 'Diluted EPS' in self.simplified_df.index else 'Basic EPS'
        if eps_metric in self.simplified_df.index:
            eps = self.simplified_df.loc[eps_metric].dropna()
            metrics['eps_years'] = len(eps)
            metrics['eps_positive_years'] = (eps > 0).sum()
            
            if len(eps) > 0:
                metrics['eps_consistency'] = (eps > 0).sum() / len(eps)
        
        # Balance sheet health
        if 'Total Assets' in self.simplified_df.index and 'Total Liabilities Net Minority Interest' in self.simplified_df.index:
            assets = self.simplified_df.loc['Total Assets'].iloc[0]
            liabilities = self.simplified_df.loc['Total Liabilities Net Minority Interest'].iloc[0]
            
            if assets > 0:
                metrics['asset_quality'] = (assets - liabilities) / assets
        
        # Profitability
        if 'Net Income' in self.simplified_df.index:
            net_income = self.simplified_df.loc['Net Income'].dropna()
            metrics['profitable_years'] = (net_income > 0).sum()
            metrics['total_years_income'] = len(net_income)
        
        return metrics
    
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
        
        # Display warnings if any
        if self.data_quality_warnings:
            print(f"\n{'='*50}")
            print("DATA QUALITY WARNINGS")
            print(f"{'='*50}")
            for warning in self.data_quality_warnings:
                print(f"⚠️  {warning}")