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
            
            if book_value_per_share < 0:
                model_warnings.append("Negative book value (liabilities exceed assets)")
            
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
            
            # Calculate growth rate (as decimal, will convert to percentage for formula)
            growth_rate_decimal = (eps_latest / eps_earliest) ** (1 / n_years) - 1
            growth_rate_percent = growth_rate_decimal * 100
            
            # Graham suggested capping growth at reasonable levels
            if growth_rate_percent > 25:
                model_warnings.append(
                    f"Growth rate ({growth_rate_percent:.1f}%) capped at 25% for Graham formula"
                )
                growth_rate_percent = 25
            
            if growth_rate_percent < 0:
                model_warnings.append("Negative growth rate; using 0% for Graham formula")
                growth_rate_percent = 0
            
            # Apply Graham Formula: V = EPS × (8.5 + 2g) × (4.4 / Y)
            # Y is bond yield as percentage (e.g., 5.21 for 5.21%)
            value = eps * (8.5 + 2 * growth_rate_percent) * (4.4 / (bond_yield * 100))
            
            self.results['graham_value'] = value
            self.confidence_scores['graham_value'] = 'Medium'
            self.model_warnings['graham_value'] = model_warnings
            
            logger.info(f"Graham Value: ${value:.2f} (Growth: {growth_rate_percent:.1f}%)")
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
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a valuation model.
        
        Args:
            model: Model to test ('dcf', 'ddm', 'graham')
            param: Parameter to vary ('discount_rate', 'terminal_growth', 'growth_rate')
            base_value: Base value for parameter (uses default if None)
            range_pct: Percentage range to test (+/- from base)
            steps: Number of steps in each direction
            
        Returns:
            DataFrame with parameter values and resulting valuations
        """
        if base_value is None:
            if param == 'discount_rate':
                base_value = DEFAULT_DISCOUNT_RATE
            elif param == 'terminal_growth':
                base_value = DEFAULT_TERMINAL_GROWTH
            elif param == 'growth_rate':
                # Use historical growth
                if model == 'dcf' and "Free Cash Flow" in self.df.index:
                    fcf = self.df.loc["Free Cash Flow"]
                    base_value = (fcf.iloc[0] / fcf.iloc[-1]) ** (1 / (len(fcf) - 1)) - 1
                else:
                    base_value = 0.10
        
        # Generate parameter range
        min_val = base_value * (1 - range_pct)
        max_val = base_value * (1 + range_pct)
        param_values = np.linspace(min_val, max_val, steps * 2 + 1)
        
        results = []
        for val in param_values:
            kwargs = {}
            if param == 'discount_rate':
                kwargs['discount_rate'] = val
            elif param == 'terminal_growth':
                kwargs['terminal_growth_rate'] = val
            elif param == 'growth_rate':
                kwargs['custom_growth_rate'] = val
            
            # Calculate valuation
            if model == 'dcf':
                intrinsic_val = self.calculate_dcf(**kwargs)
            elif model == 'ddm':
                if param == 'discount_rate':
                    kwargs = {'required_rate': val}
                intrinsic_val = self.calculate_ddm(**kwargs)
            elif model == 'graham':
                intrinsic_val = self.calculate_graham_value()
            else:
                intrinsic_val = None
            
            results.append({
                'parameter_value': val,
                'parameter_pct': val * 100 if val < 1 else val,
                'intrinsic_value': intrinsic_val,
                'change_pct': ((intrinsic_val / self.current_price - 1) * 100) if intrinsic_val and self.current_price else None
            })
        
        return pd.DataFrame(results)
    
    def calculate_all_valuations(self, **kwargs) -> Dict[str, float]:
        """
        Calculate all valuation models and return results.
        
        Args:
            **kwargs: Optional parameters to pass to individual models
            
        Returns:
            Dictionary containing all successful valuation results
        """
        logger.info("Calculating all valuation models...")
        
        self.calculate_dcf(**kwargs)
        self.calculate_ddm(**kwargs)
        self.calculate_pe_model()
        self.calculate_asset_based()
        self.calculate_graham_value(**kwargs)
        
        logger.info(f"Completed {len(self.results)} valuation models")
        return self.get_results()
    
    def get_results(self) -> Dict[str, float]:
        """Get all calculated valuation results."""
        return self.results.copy()
    
    def get_confidence_scores(self) -> Dict[str, str]:
        """Get confidence scores for each model."""
        return self.confidence_scores.copy()
    
    def get_model_warnings(self) -> Dict[str, List[str]]:
        """Get warnings for each model."""
        return self.model_warnings.copy()
    
    def get_average_valuation(self, weighted: bool = False) -> Optional[float]:
        """
        Calculate average of all valuation models.
        
        Args:
            weighted: If True, weight by confidence scores
            
        Returns:
            Average intrinsic value, or None if no valuations available
        """
        if not self.results:
            return None
        
        if not weighted or not self.confidence_scores:
            return sum(self.results.values()) / len(self.results)
        
        # Weighted average based on confidence
        weights = {'High': 3, 'Medium': 2, 'Low': 1}
        total_value = 0
        total_weight = 0
        
        for model, value in self.results.items():
            confidence = self.confidence_scores.get(model, 'Medium')
            weight = weights.get(confidence, 2)
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
            logger.warning("Current price not available for margin of safety calculation")
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
                'confidence': self.confidence_scores.get(model, 'Medium'),
                'warnings': self.model_warnings.get(model, [])
            }
        
        return analysis
    
    def get_comprehensive_report(self) -> Dict[str, any]:
        """
        Get a comprehensive valuation report.
        
        Returns:
            Dictionary with complete valuation analysis
        """
        report = {
            'valuations': self.get_results(),
            'average_valuation': self.get_average_valuation(),
            'weighted_average': self.get_average_valuation(weighted=True),
            'current_price': self.current_price,
            'confidence_scores': self.get_confidence_scores(),
            'model_warnings': self.get_model_warnings(),
            'margin_of_safety': self.get_margin_of_safety()
        }
        
        return report
    
    def print_results(self, show_margin_of_safety: bool = True, show_warnings: bool = True):
        """
        Print all valuation results in a formatted way.
        
        Args:
            show_margin_of_safety: Whether to show margin of safety analysis
            show_warnings: Whether to show model warnings
        """
        print("\n" + "="*70)
        print("INTRINSIC VALUATION RESULTS")
        print("="*70)
        
        if not self.results:
            print("No valuation results available")
            return
        
        from Config import MODEL_DISPLAY_NAMES
        
        for key, description in MODEL_DISPLAY_NAMES.items():
            if key in self.results:
                value = self.results[key]
                confidence = self.confidence_scores.get(key, 'N/A')
                print(f"{description:.<45} ${value:>10,.2f}  [{confidence}]")
        
        avg_value = self.get_average_valuation()
        weighted_avg = self.get_average_valuation(weighted=True)
        
        if avg_value:
            print(f"\n{'Average Intrinsic Value':.<45} ${avg_value:>10,.2f}")
        if weighted_avg and weighted_avg != avg_value:
            print(f"{'Weighted Average (by confidence)':.<45} ${weighted_avg:>10,.2f}")
        
        if self.current_price:
            print(f"{'Current Market Price':.<45} ${self.current_price:>10,.2f}")
        
        # Show warnings
        if show_warnings and self.model_warnings:
            has_warnings = any(warnings for warnings in self.model_warnings.values())
            if has_warnings:
                print("\n" + "="*70)
                print("MODEL WARNINGS")
                print("="*70)
                for model, warnings in self.model_warnings.items():
                    if warnings:
                        model_name = MODEL_DISPLAY_NAMES.get(model, model)
                        print(f"\n{model_name}:")
                        for warning in warnings:
                            print(f"  ⚠️  {warning}")
        
        # Margin of safety analysis
        if show_margin_of_safety and self.current_price:
            print("\n" + "="*70)
            print("MARGIN OF SAFETY ANALYSIS (Target: 25%)")
            print("="*70)
            
            margin_analysis = self.get_margin_of_safety()
            if margin_analysis:
                for model, data in margin_analysis.items():
                    model_name = MODEL_DISPLAY_NAMES.get(model, model)
                    margin_pct = data['margin_of_safety'] * 100
                    confidence = data['confidence']
                    status = "✓ UNDERVALUED" if data['is_undervalued'] else "✗ OVERVALUED"
                    
                    print(f"\n{model_name} [{confidence} Confidence]:")
                    print(f"  Intrinsic Value: ${data['intrinsic_value']:,.2f}")
                    print(f"  Margin of Safety: {margin_pct:+.1f}%  {status}")
                    print(f"  Target Buy Price: ${data['target_buy_price']:,.2f}")
        
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