import pandas as pd
from typing import Dict, Optional, Tuple
import warnings


class ValuationCalculator:
    """
    Calculates intrinsic value using multiple valuation models.
    
    Models included:
    - Discounted Cash Flow (DCF)
    - Dividend Discount Model (DDM) - Single and Multi-stage
    - P/E Multiplier Model
    - Asset-Based Valuation
    - Modern Graham Formula
    """
    
    # Default parameters
    DEFAULT_DISCOUNT_RATE = 0.10
    DEFAULT_TERMINAL_GROWTH = 0.025
    DEFAULT_PROJECTION_YEARS = 5
    DEFAULT_BOND_YIELD = 0.0521  # 5.21%
    
    def __init__(self, simplified_df: pd.DataFrame):
        """
        Initialize with simplified financial data.
        
        Args:
            simplified_df: DataFrame from IVSimplifier containing essential financial metrics
            
        Raises:
            ValueError: If simplified_df is empty or None
        """
        if simplified_df is None or simplified_df.empty:
            raise ValueError("simplified_df cannot be None or empty")
        
        self.df = simplified_df.copy()
        self.results: Dict[str, float] = {}
        self.current_price: Optional[float] = None
        
        # Try to get current price
        if "Share Price" in self.df.index:
            self.current_price = self.df.loc["Share Price"].iloc[0]
    
    def calculate_dcf(
        self, 
        discount_rate: float = DEFAULT_DISCOUNT_RATE, 
        terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH,
        projection_years: int = DEFAULT_PROJECTION_YEARS
    ) -> Optional[float]:
        """
        Calculate intrinsic value using Discounted Cash Flow model.
        
        Args:
            discount_rate: Required rate of return (default 10%)
            terminal_growth_rate: Perpetual growth rate (default 2.5%)
            projection_years: Years to project (default 5)
            
        Returns:
            Intrinsic value per share, or None if calculation fails
        """
        try:
            # Validate inputs
            if discount_rate <= terminal_growth_rate:
                warnings.warn("Discount rate must be greater than terminal growth rate")
                return None
            
            # Get Free Cash Flow data
            if "Free Cash Flow" not in self.df.index:
                warnings.warn("Free Cash Flow data not available")
                return None
            
            fcf_row = self.df.loc["Free Cash Flow"]
            fcf_latest = fcf_row.iloc[0]  # Most recent (Year 1)
            fcf_earliest = fcf_row.iloc[-1]  # Oldest year
            n_years = len(fcf_row) - 1
            
            # Validate FCF data
            if fcf_earliest <= 0 or fcf_latest <= 0:
                warnings.warn("Cannot calculate DCF: negative or zero FCF")
                return None
            
            # Calculate historical CAGR
            cagr = (fcf_latest / fcf_earliest) ** (1 / n_years) - 1
            
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
                warnings.warn("Invalid shares outstanding data")
                return None
            
            value_per_share = enterprise_value / shares_outstanding
            
            self.results['dcf'] = value_per_share
            return value_per_share
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"DCF calculation failed: {e}")
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
        try:
            # Validate inputs
            if required_rate <= terminal_growth:
                warnings.warn("Required rate must be greater than terminal growth")
                return None
            
            # Get dividend data
            if "Annual Dividends" not in self.df.index:
                warnings.warn("Dividend data not available")
                return None
            
            dividends_row = self.df.loc["Annual Dividends"]
            
            # Check if company pays dividends
            if dividends_row.sum() == 0:
                warnings.warn("Company does not pay dividends")
                return None
            
            div_latest = dividends_row.iloc[0]  # Most recent
            div_earliest = dividends_row.iloc[-1]  # Oldest
            n_years = len(dividends_row) - 1
            
            if div_earliest <= 0 or div_latest <= 0:
                warnings.warn("Cannot calculate DDM: negative or zero dividends")
                return None
            
            # Calculate dividend growth rate
            growth_rate = (div_latest / div_earliest) ** (1 / n_years) - 1
            
            # Single-stage Gordon Growth Model
            dividend_next_year = div_latest * (1 + growth_rate)
            single_stage_value = dividend_next_year / (required_rate - growth_rate)
            
            # Multi-stage DDM
            pv_dividends = 0
            for t in range(1, projection_years + 1):
                div_projected = div_latest * (1 + growth_rate) ** t
                pv_dividends += div_projected / (1 + required_rate) ** t
            
            # Terminal value
            terminal_dividend = div_latest * (1 + growth_rate) ** projection_years * (1 + terminal_growth)
            pv_terminal = (terminal_dividend / (required_rate - terminal_growth)) / (1 + required_rate) ** projection_years
            
            multi_stage_value = pv_dividends + pv_terminal
            
            self.results['ddm_single_stage'] = single_stage_value
            self.results['ddm_multi_stage'] = multi_stage_value
            return multi_stage_value
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"DDM calculation failed: {e}")
            return None
    
    def calculate_pe_model(self) -> Optional[float]:
        """
        Calculate intrinsic value using P/E Multiplier model.
        Uses average historical P/E ratio multiplied by current EPS.
        
        Returns:
            Intrinsic value per share, or None if calculation fails
        """
        try:
            # Get EPS data
            eps = self._get_eps()
            if eps is None or eps <= 0:
                warnings.warn("Invalid EPS data for P/E model")
                return None
            
            # Get historical P/E ratios
            if "P/E Ratio" not in self.df.index:
                warnings.warn("P/E Ratio data not available")
                return None
            
            pe_row = self.df.loc["P/E Ratio"].dropna()
            
            if len(pe_row) == 0:
                warnings.warn("No valid P/E ratios available")
                return None
            
            expected_pe = pe_row.mean()
            
            value = eps * expected_pe
            self.results['pe_model'] = value
            return value
            
        except (KeyError, IndexError) as e:
            warnings.warn(f"P/E Model calculation failed: {e}")
            return None
    
    def calculate_asset_based(self) -> Optional[float]:
        """
        Calculate intrinsic value using Asset-Based valuation (Book Value).
        
        Returns:
            Book value per share, or None if calculation fails
        """
        try:
            # Get balance sheet data
            if "Total Assets" not in self.df.index:
                warnings.warn("Total Assets data not available")
                return None
            
            if "Total Liabilities Net Minority Interest" not in self.df.index:
                warnings.warn("Total Liabilities data not available")
                return None
            
            total_assets = self.df.loc["Total Assets"].iloc[0]
            total_liabilities = self.df.loc["Total Liabilities Net Minority Interest"].iloc[0]
            shares_outstanding = self._get_shares_outstanding()
            
            if shares_outstanding is None or shares_outstanding <= 0:
                warnings.warn("Invalid shares outstanding data")
                return None
            
            book_value_per_share = (total_assets - total_liabilities) / shares_outstanding
            self.results['asset_based'] = book_value_per_share
            return book_value_per_share
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"Asset-Based valuation failed: {e}")
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
        try:
            # Get EPS data
            eps = self._get_eps()
            if eps is None or eps <= 0:
                warnings.warn("Invalid EPS data for Graham formula")
                return None
            
            # Calculate EPS growth rate
            if "Diluted EPS" not in self.df.index:
                warnings.warn("EPS data not available for growth calculation")
                return None
            
            eps_row = self.df.loc["Diluted EPS"].dropna()
            if len(eps_row) < 2:
                warnings.warn("Insufficient EPS history for growth calculation")
                return None
            
            eps_latest = eps_row.iloc[0]
            eps_earliest = eps_row.iloc[-1]
            n_years = len(eps_row) - 1
            
            if eps_earliest <= 0:
                warnings.warn("Cannot calculate growth: negative or zero historical EPS")
                return None
            
            # Calculate growth rate in percentage
            growth_rate = ((eps_latest / eps_earliest) ** (1 / n_years) - 1) * 100
            
            # Apply Graham Formula
            value = eps * (8.5 + 2 * growth_rate) * (4.4 / (bond_yield * 100))
            
            self.results['graham_value'] = value
            return value
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"Graham Value calculation failed: {e}")
            return None
    
    def calculate_all_valuations(self, **kwargs) -> Dict[str, float]:
        """
        Calculate all valuation models and return results.
        
        Args:
            **kwargs: Optional parameters to pass to individual models
            
        Returns:
            Dictionary containing all successful valuation results
        """
        self.calculate_dcf(**kwargs)
        self.calculate_ddm(**kwargs)
        self.calculate_pe_model()
        self.calculate_asset_based()
        self.calculate_graham_value(**kwargs)
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, float]:
        """Get all calculated valuation results."""
        return self.results.copy()
    
    def get_average_valuation(self) -> Optional[float]:
        """
        Calculate average of all valuation models.
        
        Returns:
            Average intrinsic value, or None if no valuations available
        """
        if not self.results:
            return None
        return sum(self.results.values()) / len(self.results)
    
    def get_margin_of_safety(self, target_margin: float = 0.25) -> Optional[Dict[str, any]]:
        """
        Calculate margin of safety for each valuation method.
        
        Args:
            target_margin: Desired margin of safety (default 25%)
            
        Returns:
            Dictionary with margin of safety analysis
        """
        if self.current_price is None:
            warnings.warn("Current price not available")
            return None
        
        if not self.results:
            warnings.warn("No valuation results available")
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
                'target_buy_price': buy_price
            }
        
        return analysis
    
    def print_results(self, show_margin_of_safety: bool = True):
        """
        Print all valuation results in a formatted way.
        
        Args:
            show_margin_of_safety: Whether to show margin of safety analysis
        """
        print("\n" + "="*60)
        print("INTRINSIC VALUATION RESULTS")
        print("="*60)
        
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
                print(f"{description:.<50} ${value:,.2f}")
        
        avg_value = self.get_average_valuation()
        if avg_value:
            print(f"\n{'Average Intrinsic Value':.<50} ${avg_value:,.2f}")
        
        if self.current_price:
            print(f"{'Current Market Price':.<50} ${self.current_price:,.2f}")
        
        # Margin of safety analysis
        if show_margin_of_safety and self.current_price:
            print("\n" + "="*60)
            print("MARGIN OF SAFETY ANALYSIS")
            print("="*60)
            
            margin_analysis = self.get_margin_of_safety()
            if margin_analysis:
                for model, data in margin_analysis.items():
                    model_name = models.get(model, model)
                    margin_pct = data['margin_of_safety'] * 100
                    status = "✓ UNDERVALUED" if data['is_undervalued'] else "✗ OVERVALUED"
                    
                    print(f"\n{model_name}:")
                    print(f"  Margin of Safety: {margin_pct:+.1f}% {status}")
                    print(f"  Target Buy Price (25% margin): ${data['target_buy_price']:,.2f}")
        
        print("="*60)
    
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