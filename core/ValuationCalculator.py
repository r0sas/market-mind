import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import logging
from core.Config import (
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
                self.model_warnings['asset_based'] = model_warnings
                return None
            
            total_assets = self.df.loc["Total Assets"].iloc[0]
            total_liabilities = self.df.loc["Total Liabilities Net Minority Interest"].iloc[0]
            shares_outstanding = self._get_shares_outstanding()
            
            if shares_outstanding is None or shares_outstanding <= 0:
                model_warnings.append("Invalid shares outstanding data")
                self.model_warnings['asset_based'] = model_warnings
                return None
            
            book_value_per_share = (total_assets - total_liabilities) / shares_outstanding
            
            if book_value_per_share <= 0:
                model_warnings.append("Book value is negative (liabilities exceed assets)")
            
            self.results['asset_based'] = book_value_per_share
            self.confidence_scores['asset_based'] = 'Medium'
            self.model_warnings['asset_based'] = model_warnings
            
            logger.info(f"Asset-Based: ${book_value_per_share:.2f}")
            return book_value_per_share
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            model_warnings.append(f"Asset-Based valuation failed: {str(e)}")
            self.model_warnings['asset_based'] = model_warnings
            return None
    
    def calculate_graham_value(self, bond_yield: float = DEFAULT_BOND_YIELD) -> Optional[float]:
        """
        Calculate intrinsic value using Modern Graham Formula.
        Formula: V = EPS × (8.5 + 2g) × (4.4 / Y)
        
        Args:
            bond_yield: Corporate bond yield (default 5.21%)
            
        Returns:
            Graham value per share, or None if calculation fails
        """
        model_warnings = []
        
        try:
            # Get EPS data
            eps = self._get_eps()
            if eps is None or eps <= 0:
                model_warnings.append("Invalid EPS data for Graham formula")
                self.model_warnings['graham_value'] = model_warnings
                return None
            
            # Calculate EPS growth rate
            if "Diluted EPS" not in self.df.index:
                model_warnings.append("EPS data not available for growth calculation")
                self.model_warnings['graham_value'] = model_warnings
                return None
            
            eps_row = self.df.loc["Diluted EPS"].dropna()
            if len(eps_row) < 2:
                model_warnings.append("Insufficient EPS history for growth calculation")
                self.model_warnings['graham_value'] = model_warnings
                return None
            
            eps_latest = eps_row.iloc[0]
            eps_earliest = eps_row.iloc[-1]
            n_years = len(eps_row) - 1
            
            if eps_earliest <= 0:
                model_warnings.append("Cannot calculate growth: negative or zero historical EPS")
                self.model_warnings['graham_value'] = model_warnings
                return None
            
            # Calculate growth rate in percentage
            growth_rate_decimal = (eps_latest / eps_earliest) ** (1 / n_years) - 1
            growth_rate_pct = growth_rate_decimal * 100
            
            # Graham suggested capping growth at reasonable levels
            if growth_rate_pct > 20:
                model_warnings.append(f"Growth rate ({growth_rate_pct:.1f}%) capped at 20% for Graham formula")
                growth_rate_pct = 20
            
            # Apply Graham Formula
            value = eps * (8.5 + 2 * growth_rate_pct) * (4.4 / (bond_yield * 100))
            
            self.results['graham_value'] = value
            self.confidence_scores['graham_value'] = 'Medium'
            self.model_warnings['graham_value'] = model_warnings
            
            logger.info(f"Graham Value: ${value:.2f} (Growth: {growth_rate_pct:.1f}%)")
            return value
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            model_warnings.append(f"Graham Value calculation failed: {str(e)}")
            self.model_warnings['graham_value'] = model_warnings
            return None
    
    def sensitivity_analysis(
        self,
        model: str = 'dcf',
        param: str = 'discount_rate',
        base_value: Optional[float] = None,
        range_pct: float = 0.2,
        steps: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis on a valuation model.
        
        Args:
            model: Model to analyze ('dcf', 'ddm_multi_stage', 'graham_value')
            param: Parameter to vary ('discount_rate', 'terminal_growth', 'growth_rate')
            base_value: Base value for parameter (uses default if None)
            range_pct: Percentage range to vary parameter (default 20%)
            steps: Number of steps in each direction (default 5)
            
        Returns:
            Dictionary with parameter values and corresponding valuations
        """
        if base_value is None:
            base_value = DEFAULT_DISCOUNT_RATE if param == 'discount_rate' else DEFAULT_TERMINAL_GROWTH
        
        # Generate parameter range
        param_values = np.linspace(
            base_value * (1 - range_pct),
            base_value * (1 + range_pct),
            steps * 2 + 1
        )
        
        valuations = []
        
        for param_val in param_values:
            if model == 'dcf':
                if param == 'discount_rate':
                    val = self.calculate_dcf(discount_rate=param_val)
                elif param == 'terminal_growth':
                    val = self.calculate_dcf(terminal_growth_rate=param_val)
                else:
                    val = self.calculate_dcf(custom_growth_rate=param_val)
            elif model == 'ddm_multi_stage':
                if param == 'discount_rate':
                    val = self.calculate_ddm(required_rate=param_val)
                else:
                    val = self.calculate_ddm(terminal_growth=param_val)
            else:
                val = None
            
            valuations.append(val if val is not None else 0)
        
        return {
            'parameter': param,
            'values': param_values.tolist(),
            'valuations': valuations
        }
    
    def calculate_all_valuations(
        self, 
        models_to_calculate: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate all valuation models and return results.
        
        Args:
            models_to_calculate: List of specific models to calculate.
                               If None, calculates all models.
                               Valid values: 'dcf', 'ddm_single_stage', 'ddm_multi_stage',
                                           'pe_model', 'graham_value', 'asset_based'
            **kwargs: Optional parameters to pass to individual models
                     (discount_rate, terminal_growth_rate, bond_yield)
            
        Returns:
            Dictionary containing all successful valuation results
        """
        logger.info("Calculating valuation models...")
        
        # If no specific models requested, calculate all
        if models_to_calculate is None:
            models_to_calculate = [
                'dcf', 'ddm_single_stage', 'ddm_multi_stage',
                'pe_model', 'graham_value', 'asset_based'
            ]
        
        logger.info(f"Models to calculate: {models_to_calculate}")
        
        # Extract parameters with defaults
        discount_rate = kwargs.get('discount_rate', DEFAULT_DISCOUNT_RATE)
        terminal_growth = kwargs.get('terminal_growth_rate', DEFAULT_TERMINAL_GROWTH)
        bond_yield = kwargs.get('bond_yield', DEFAULT_BOND_YIELD)
        
        # Calculate each requested model
        if 'dcf' in models_to_calculate:
            self.calculate_dcf(
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth
            )
        
        # DDM models (single or multi-stage)
        if 'ddm_single_stage' in models_to_calculate or 'ddm_multi_stage' in models_to_calculate:
            self.calculate_ddm(
                required_rate=discount_rate,
                terminal_growth=terminal_growth
            )
        
        if 'pe_model' in models_to_calculate:
            self.calculate_pe_model()
        
        if 'asset_based' in models_to_calculate:
            self.calculate_asset_based()
        
        if 'graham_value' in models_to_calculate:
            self.calculate_graham_value(bond_yield=bond_yield)
        
        logger.info(f"Completed {len(self.results)} valuation models")
        return self.get_results()
    
    def get_results(self) -> Dict[str, float]:
        """Get all calculated valuation results."""
        return self.results.copy()
    
    def get_confidence_scores(self) -> Dict[str, str]:
        """Get confidence scores for all calculated models."""
        return self.confidence_scores.copy()
    
    def get_model_warnings(self) -> Dict[str, List[str]]:
        """Get warnings for all models."""
        return self.model_warnings.copy()
    
    def get_average_valuation(self, weighted: bool = False) -> Optional[float]:
        """
        Calculate average of all valuation models.
        
        Args:
            weighted: If True, weight by confidence scores (High=3, Medium=2, Low=1)
            
        Returns:
            Average intrinsic value, or None if no valuations available
        """
        if not self.results:
            return None
        
        if not weighted:
            return sum(self.results.values()) / len(self.results)
        
        # Weighted average
        confidence_weights = {'High': 3, 'Medium': 2, 'Low': 1}
        total_value = 0
        total_weight = 0
        
        for model, value in self.results.items():
            weight = confidence_weights.get(self.confidence_scores.get(model, 'Medium'), 2)
            total_value += value * weight
            total_weight += weight
        
        return total_value / total_weight if total_weight > 0 else None
    
    def get_margin_of_safety(
        self, 
        target_margin: float = DEFAULT_MARGIN_OF_SAFETY
    ) -> Optional[Dict[str, any]]:
        """
        Calculate margin of safety for each valuation method.
        
        Args:
            target_margin: Desired margin of safety (default 25%)
            
        Returns:
            Dictionary with margin of safety analysis
        """
        if self.current_price is None:
            logger.warning("Current price not available")
            return None
        
        if not self.results:
            logger.warning("No valuation results available")
            return None
        
        analysis = {}
        for model, intrinsic_value in self.results.items():
            margin = (intrinsic_value - self.current_price) / intrinsic_value
            buy_price = intrinsic_value * (1 - target_margin)
            
            analysis[model] = {
                'intrinsic_value': intrinsic_value,
                'current_price': self.current_price,
                'margin_of_safety': margin,
                'is_undervalued': margin >= target_margin,
                'target_buy_price': buy_price,
                'confidence': self.confidence_scores.get(model, 'N/A'),
                'warnings': self.model_warnings.get(model, [])
            }
        
        return analysis
    
    def print_results(self, show_margin_of_safety: bool = True, show_confidence: bool = True):
        """
        Print all valuation results in a formatted way.
        
        Args:
            show_margin_of_safety: Whether to show margin of safety analysis
            show_confidence: Whether to show confidence scores
        """
        print("\n" + "="*70)
        print("INTRINSIC VALUATION RESULTS")
        print("="*70)
        
        if not self.results:
            print("No valuation results available")
            return
        
        models = {
            'dcf': 'DCF Intrinsic Value per Share',
            'ddm_single_stage': 'DDM Single-Stage Value per Share', 
            'ddm_multi_stage': 'DDM Multi-Stage Value per Share',
            'pe_model': 'P/E Model Intrinsic Value',
            'asset_based': 'Asset-Based Value per Share',
            'graham_value': 'Modern Graham Value per Share'
        }
        
        for key, description in models.items():
            if key in self.results:
                value = self.results[key]
                confidence = self.confidence_scores.get(key, 'N/A')
                
                if show_confidence:
                    print(f"{description:.<45} ${value:>10,.2f}  [{confidence}]")
                else:
                    print(f"{description:.<50} ${value:>10,.2f}")
        
        avg_value = self.get_average_valuation()
        weighted_avg = self.get_average_valuation(weighted=True)
        
        if avg_value:
            print(f"\n{'Simple Average Intrinsic Value':.<50} ${avg_value:>10,.2f}")
        if weighted_avg:
            print(f"{'Confidence-Weighted Average':.<50} ${weighted_avg:>10,.2f}")
        
        if self.current_price:
            print(f"{'Current Market Price':.<50} ${self.current_price:>10,.2f}")
        
        # Show warnings
        if show_confidence and any(self.model_warnings.values()):
            print("\n" + "="*70)
            print("MODEL WARNINGS")
            print("="*70)
            for model, warnings in self.model_warnings.items():
                if warnings:
                    model_name = models.get(model, model)
                    print(f"\n{model_name}:")
                    for warning in warnings:
                        print(f"  ⚠️  {warning}")
        
        # Margin of safety analysis
        if show_margin_of_safety and self.current_price:
            print("\n" + "="*70)
            print("MARGIN OF SAFETY ANALYSIS")
            print("="*70)
            
            margin_analysis = self.get_margin_of_safety()
            if margin_analysis:
                for model, data in margin_analysis.items():
                    model_name = models.get(model, model)
                    margin_pct = data['margin_of_safety'] * 100
                    status = "✓ UNDERVALUED" if data['is_undervalued'] else "✗ OVERVALUED"
                    confidence = data['confidence']
                    
                    print(f"\n{model_name} [{confidence}]:")
                    print(f"  Margin of Safety: {margin_pct:+.1f}% {status}")
                    print(f"  Target Buy Price (25% margin): ${data['target_buy_price']:,.2f}")
        
        print("="*70)
    
    def _get_eps(self) -> Optional[float]:
        """Helper method to get the most recent EPS (prefer Diluted over Basic)"""
        if "Diluted EPS" in self.df.index:
            return self.df.loc["Diluted EPS"].iloc[0]
        elif "Basic EPS" in self.df.index:
            return self.df.loc["Basic EPS"].iloc[0]
        return None
    
    def _get_shares_outstanding(self) -> Optional[float]:
        """Helper method to get shares outstanding (prefer Basic over Diluted)"""
        if "Basic Average Shares" in self.df.index:
            return self.df.loc["Basic Average Shares"].iloc[0]
        elif "Diluted Average Shares" in self.df.index:
            return self.df.loc["Diluted Average Shares"].iloc[0]
        return None