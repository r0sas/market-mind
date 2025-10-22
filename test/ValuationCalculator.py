import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import logging
from test.Config import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_PROJECTION_YEARS,
    DEFAULT_BOND_YIELD,
    DEFAULT_MARGIN_OF_SAFETY,
    MAX_REASONABLE_GROWTH_RATE,
    MAX_REASONABLE_PE,
    CONFIDENCE_THRESHOLDS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValuationError(Exception):
    """Custom exception for valuation errors"""
    pass


class ValuationCalculator:
    """
    Calculates intrinsic value using multiple valuation models.
    
    Models included:
    - Discounted Cash Flow (DCF)
    - Dividend Discount Model (DDM) - Single and Multi-stage
    - P/E Multiplier Model
    - Asset-Based Valuation
    - Modern Graham Formula
    
    Example:
        >>> calculator = ValuationCalculator(simplified_df)
        >>> calculator.calculate_all_valuations()
        >>> results = calculator.get_results()
    """
    
    def __init__(self, simplified_df: pd.DataFrame):
        """
        Initialize with simplified financial data.
        
        Args:
            simplified_df: DataFrame from IVSimplifier containing essential financial metrics
            
        Raises:
            ValuationError: If simplified_df is empty or None
        """
        if simplified_df is None or simplified_df.empty:
            raise ValuationError("simplified_df cannot be None or empty")
        
        self.df = simplified_df.copy()
        self.results: Dict[str, float] = {}
        self.confidence_scores: Dict[str, str] = {}
        self.model_warnings: Dict[str, List[str]] = {}
        self.current_price: Optional[float] = None
        
        # Try to get current price
        if "Share Price" in self.df.index:
            self.current_price = self.df.loc["Share Price"].iloc[0]
    
    def calculate_dcf(
        self, 
        discount_rate: float = DEFAULT_DISCOUNT_RATE, 
        terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH,
        projection_years: int = DEFAULT_PROJECTION_YEARS,
        custom_growth_rate: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate intrinsic value using Discounted Cash Flow model.
        
        Args:
            discount_rate: Required rate of return (default 10%)
            terminal_growth_rate: Perpetual growth rate (default 2.5%)
            projection_years: Years to project (default 5)
            custom_growth_rate: Optional custom growth rate instead of historical CAGR
            
        Returns:
            Intrinsic value per share, or None if calculation fails
        """
        model_warnings = []
        
        try:
            # Validate inputs
            if discount_rate <= terminal_growth_rate:
                model_warnings.append("Discount rate must be greater than terminal growth rate")
                self.model_warnings['dcf'] = model_warnings
                return None
            
            # Get Free Cash Flow data
            if "Free Cash Flow" not in self.df.index:
                model_warnings.append("Free Cash Flow data not available")
                self.model_warnings['dcf'] = model_warnings
                return None
            
            fcf_row = self.df.loc["Free Cash Flow"]
            fcf_latest = fcf_row.iloc[0]  # Most recent (Year 1)
            fcf_earliest = fcf_row.iloc[-1]  # Oldest year
            n_years = len(fcf_row) - 1
            
            # Validate FCF data
            if fcf_earliest <= 0 or fcf_latest <= 0:
                model_warnings.append("Cannot calculate DCF: negative or zero FCF")
                self.model_warnings['dcf'] = model_warnings
                return None
            
            # Calculate historical CAGR or use custom rate
            if custom_growth_rate is not None:
                cagr = custom_growth_rate
                model_warnings.append(f"Using custom growth rate: {cagr*100:.1f}%")
            else:
                cagr = (fcf_latest / fcf_earliest) ** (1 / n_years) - 1
            
            # Validate growth rate is reasonable
            if cagr > MAX_REASONABLE_GROWTH_RATE:
                model_warnings.append(
                    f"Warning: Growth rate ({cagr*100:.1f}%) is very high. "
                    f"Consider using a more conservative estimate."
                )
            
            # Calculate FCF volatility for confidence scoring
            fcf_cv = fcf_row.std() / fcf_row.mean()
            
            # Project future FCFs and calculate present values
            present_value_fcf = []
            
            for year in range(1, projection_years + 1):
                projected_fcf = fcf_latest * (1 + cagr) ** year
                pv_fcf = projected_fcf / (1 + discount_rate) ** year
                present_value_fcf.append(pv_fcf)
            
            # Calculate terminal value
            final_fcf = fcf_latest * (1 + cagr) ** projection_years
            terminal_value = (final_fcf * (1 + terminal_growth_rate) / 
                            (discount_rate - terminal_growth_rate))
            pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
            
            # Calculate enterprise value
            total_pv_fcf = sum(present_value_fcf)
            enterprise_value = total_pv_fcf + pv_terminal
            
            # Convert to per-share value
            shares_outstanding = self._get_shares_outstanding()
            if shares_outstanding is None or shares_outstanding <= 0:
                model_warnings.append("Invalid shares outstanding data")
                self.model_warnings['dcf'] = model_warnings
                return None
            
            value_per_share = enterprise_value / shares_outstanding
            
            # Determine confidence score
            if fcf_cv > CONFIDENCE_THRESHOLDS['fcf_volatility_high']:
                confidence = 'Low'
            elif fcf_cv > CONFIDENCE_THRESHOLDS['fcf_volatility_medium']:
                confidence = 'Medium'
            else:
                confidence = 'High'
            
            self.confidence_scores['dcf'] = confidence
            self.results['dcf'] = value_per_share
            self.model_warnings['dcf'] = model_warnings
            
            logger.info(f"DCF: ${value_per_share:.2f} (Confidence: {confidence}, Growth: {cagr*100:.1f}%)")
            return value_per_share
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            model_warnings.append(f"DCF calculation failed: {str(e)}")
            self.model_warnings['dcf'] = model_warnings
            return None
    
    def calculate_ddm(
        self, 
        required_rate: float = DEFAULT_DISCOUNT_RATE, 
        terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
        projection_years: int = DEFAULT_PROJECTION_YEARS
    ) -> Optional[float]:
        """
        Calculate intrinsic value using Dividend Discount Model (multi-stage).
        
        Args:
            required_rate: Required rate of return (default 10%)
            terminal_growth: Terminal growth rate (default 2.5%)
            projection_years: Years to project (default 5)
            
        Returns:
            Intrinsic value per share, or None if calculation fails
        """
        model_warnings = []
        
        try:
            # Validate inputs
            if required_rate <= terminal_growth:
                model_warnings.append("Required rate must be greater than terminal growth")
                self.model_warnings['ddm_multi_stage'] = model_warnings
                return None
            
            # Get dividend data
            if "Annual Dividends" not in self.df.index:
                model_warnings.append("Dividend data not available")
                self.model_warnings['ddm_multi_stage'] = model_warnings
                return None
            
            dividends_row = self.df.loc["Annual Dividends"]
            
            # Check if company pays dividends
            if dividends_row.sum() == 0:
                model_warnings.append("Company does not pay dividends")
                self.model_warnings['ddm_single_stage'] = model_warnings
                self.model_warnings['ddm_multi_stage'] = model_warnings
                return None
            
            div_latest = dividends_row.iloc[0]  # Most recent
            div_earliest = dividends_row.iloc[-1]  # Oldest
            n_years = len(dividends_row) - 1
            
            if div_earliest <= 0 or div_latest <= 0:
                model_warnings.append("Cannot calculate DDM: negative or zero dividends in history")
                self.model_warnings['ddm_multi_stage'] = model_warnings
                return None
            
            # Calculate dividend growth rate
            growth_rate = (div_latest / div_earliest) ** (1 / n_years) - 1
            
            # Check for reasonable growth
            if growth_rate > MAX_REASONABLE_GROWTH_RATE:
                model_warnings.append(
                    f"Warning: Dividend growth rate ({growth_rate*100:.1f}%) is very high"
                )
            
            # Single-stage Gordon Growth Model
            dividend_next_year = div_latest * (1 + growth_rate)
            try:
                single_stage_value = dividend_next_year / (required_rate - growth_rate)
                self.results['ddm_single_stage'] = single_stage_value
                self.confidence_scores['ddm_single_stage'] = 'Medium'
            except ZeroDivisionError:
                model_warnings.append("Single-stage DDM: growth rate equals required rate")
                single_stage_value = None
            
            # Multi-stage DDM
            pv_dividends = 0
            for t in range(1, projection_years + 1):
                div_projected = div_latest * (1 + growth_rate) ** t
                pv_dividends += div_projected / (1 + required_rate) ** t
            
            # Terminal value
            terminal_dividend = div_latest * (1 + growth_rate) ** projection_years * (1 + terminal_growth)
            pv_terminal = (terminal_dividend / (required_rate - terminal_growth)) / (1 + required_rate) ** projection_years
            
            multi_stage_value = pv_dividends + pv_terminal
            
            self.results['ddm_multi_stage'] = multi_stage_value
            self.confidence_scores['ddm_multi_stage'] = 'Medium'
            self.model_warnings['ddm_single_stage'] = model_warnings
            self.model_warnings['ddm_multi_stage'] = model_warnings
            
            logger.info(f"DDM Multi-Stage: ${multi_stage_value:.2f} (Growth: {growth_rate*100:.1f}%)")
            return multi_stage_value
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            model_warnings.append(f"DDM calculation failed: {str(e)}")
            self.model_warnings['ddm_multi_stage'] = model_warnings
            return None
    
    def calculate_pe_model(self, use_median: bool = True) -> Optional[float]:
        """
        Calculate intrinsic value using P/E Multiplier model.
        Uses historical P/E ratio (median or mean) multiplied by current EPS.
        
        Args:
            use_median: If True, use median P/E; if False, use mean (default: True)
            
        Returns:
            Intrinsic value per share, or None if calculation fails
        """
        model_warnings = []
        
        try:
            # Get EPS data
            eps = self._get_eps()
            if eps is None or eps <= 0:
                model_warnings.append("Invalid EPS data for P/E model")
                self.model_warnings['pe_model'] = model_warnings
                return None
            
            # Get historical P/E ratios
            if "P/E Ratio" not in self.df.index:
                model_warnings.append("P/E Ratio data not available")
                self.model_warnings['pe_model'] = model_warnings
                return None
            
            pe_row = self.df.loc["P/E Ratio"].dropna()
            
            if len(pe_row) == 0:
                model_warnings.append("No valid P/E ratios available")
                self.model_warnings['pe_model'] = model_warnings
                return None
            
            # Filter out extreme outliers (negative or > MAX_REASONABLE_PE)
            pe_filtered = pe_row[(pe_row > 0) & (pe_row < MAX_REASONABLE_PE)]
            
            if len(pe_filtered) == 0:
                model_warnings.append("All P/E ratios are outliers or negative")
                self.model_warnings['pe_model'] = model_warnings
                return None
            
            if len(pe_filtered) < len(pe_row):
                removed = len(pe_row) - len(pe_filtered)
                model_warnings.append(f"Removed {removed} outlier P/E value(s)")
            
            # Calculate expected P/E
            if use_median:
                expected_pe = pe_filtered.median()
            else:
                expected_pe = pe_filtered.mean()
            
            value = eps * expected_pe
            
            # Confidence based on P/E volatility
            pe_cv = pe_filtered.std() / pe_filtered.mean()
            if pe_cv > 0.5:
                confidence = 'Low'
            elif pe_cv > 0.3:
                confidence = 'Medium'
            else:
                confidence = 'High'
            
            self.results['pe_model'] = value
            self.confidence_scores['pe_model'] = confidence
            self.model_warnings['pe_model'] = model_warnings
            
            logger.info(f"P/E Model: ${value:.2f} (Expected P/E: {expected_pe:.1f}, Confidence: {confidence})")
            return value
            
        except (KeyError, IndexError) as e:
            model_warnings.append(f"P/E Model calculation failed: {str(e)}")
            self.model_warnings['pe_model'] = model_warnings
            return None
    
    def calculate_asset_based(self) -> Optional[float]:
        """
        Calculate intrinsic value using Asset-Based valuation (Book Value).
        
        Returns:
            Book value per share, or None if calculation fails
        """
        model_warnings = []
        
        try:
            # Get balance sheet data
            if "Total Assets" not in self.df.index:
                model_warnings.append("Total Assets data not available")
                self.model_warnings['asset_based'] = model_warnings
                return None
            
            if "Total Liabilities Net Minority Interest" not in self.df.index:
                model_warnings.append("Total Liabilities data not available")
                self.model_warnings['asset_based'] = model