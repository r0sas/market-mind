import pandas as pd
from typing import Dict, Optional, Tuple
import warnings

class ValuationCalculator:
    """
    Calculates intrinsic value using multiple valuation models.
    
    Models included:
    - Discounted Cash Flow (DCF)
    - Dividend Discount Model (DDM)
    - P/E Multiplier Model
    - Asset-Based Valuation
    - Modern Graham Formula
    """
    
    def __init__(self, simplified_df: pd.DataFrame):
        """
        Initialize with simplified financial data.
        
        Args:
            simplified_df: DataFrame from IV_simplifier containing essential financial metrics
        """
        self.df = simplified_df.copy()
        self.results = {}
    
    def calculate_dcf(self, discount_rate: float = 0.10, 
                     terminal_growth_rate: float = 0.025,
                     projection_years: int = 5) -> Optional[float]:
        """
        Calculate intrinsic value using Discounted Cash Flow model.
        """
        try:
            # Get Free Cash Flow data
            fcf_row = self.df.loc["Free Cash Flow"]
            FCFL = fcf_row.iloc[0]  # Latest FCF
            FCFE = fcf_row.iloc[-1] # Earliest FCF
            N = len(fcf_row) - 1
            
            # Validate FCF data
            if FCFE <= 0 or FCFL <= 0:
                warnings.warn("Cannot calculate DCF: negative or zero FCF.")
                return None
            
            # Calculate CAGR
            CAGR = (FCFL / FCFE) ** (1 / N) - 1
            
            # Project future FCFs
            future_fcf = []
            present_value_fcf = []
            
            for year in range(1, projection_years + 1):
                fcf = FCFE * (1 + CAGR) ** year
                pv_fcf = fcf / (1 + discount_rate) ** year
                future_fcf.append(fcf)
                present_value_fcf.append(pv_fcf)
            
            # Calculate terminal value
            terminal_value = (future_fcf[-1] * (1 + terminal_growth_rate) / 
                            (discount_rate - terminal_growth_rate))
            pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
            
            # Calculate enterprise value and per-share value
            total_pv_fcf = sum(present_value_fcf)
            enterprise_value = total_pv_fcf + pv_terminal
            
            shares_outstanding = self.df.loc["Basic Average Shares"].iloc[-1]
            value_per_share = enterprise_value / shares_outstanding
            
            self.results['dcf'] = value_per_share
            return value_per_share
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"DCF calculation failed: {e}")
            return None
    
    def calculate_ddm(self, required_rate: float = 0.10, 
                     terminal_growth: float = 0.03,
                     projection_years: int = 5) -> Optional[float]:
        """
        Calculate intrinsic value using Dividend Discount Model.
        """
        try:
            dividends_row = self.df.loc["Dividends"]
            
            # Check if company pays dividends
            if dividends_row.sum() == 0:
                warnings.warn("Company does not pay dividends")
                return None
            
            D_earliest = dividends_row.iloc[0]
            D_latest = dividends_row.iloc[-1]
            n_years = len(dividends_row) - 1
            
            # Calculate dividend growth rate
            growth_rate = (D_latest / D_earliest) ** (1 / n_years) - 1
            
            # Gordon Growth Model (single-stage)
            dividend_next_year = D_latest * (1 + growth_rate)
            intrinsic_value = dividend_next_year / (required_rate - growth_rate)
            
            # Multi-stage DDM
            dividends_projection = [
                D_latest * (1 + growth_rate) ** t 
                for t in range(1, projection_years + 1)
            ]
            
            pv_dividends = sum(
                div / (1 + required_rate) ** t 
                for t, div in enumerate(dividends_projection, 1)
            )
            
            terminal_dividend = dividends_projection[-1] * (1 + terminal_growth)
            pv_terminal = (terminal_dividend / 
                          ((required_rate - terminal_growth) * 
                           (1 + required_rate) ** projection_years))
            
            multi_stage_value = pv_dividends + pv_terminal
            
            self.results['ddm_single_stage'] = intrinsic_value
            self.results['ddm_multi_stage'] = multi_stage_value
            return multi_stage_value
            
        except (KeyError, ZeroDivisionError) as e:
            warnings.warn(f"DDM calculation failed: {e}")
            return None
    
    def calculate_pe_model(self) -> Optional[float]:
        """Calculate intrinsic value using P/E Multiplier model."""
        try:
            EPS = self.df.loc["Diluted EPS"].iloc[0]
            pe_row = self.df.loc["P/E Ratio"].dropna()
            expected_PE = pe_row.mean()
            
            value = EPS * expected_PE
            self.results['pe_model'] = value
            return value
            
        except (KeyError, IndexError) as e:
            warnings.warn(f"P/E Model calculation failed: {e}")
            return None
    
    def calculate_asset_based(self) -> Optional[float]:
        """Calculate intrinsic value using Asset-Based valuation."""
        try:
            total_assets = self.df.loc["Total Assets"].iloc[0]
            total_liabilities = self.df.loc["Total Liabilities Net Minority Interest"].iloc[0]
            shares_outstanding = self.df.loc["Basic Average Shares"].iloc[0]
            
            value_per_share = (total_assets - total_liabilities) / shares_outstanding
            self.results['asset_based'] = value_per_share
            return value_per_share
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"Asset-Based valuation failed: {e}")
            return None
    
    def calculate_graham_value(self, bond_yield: float = 5.21) -> Optional[float]:
        """Calculate intrinsic value using Modern Graham Formula."""
        try:
            EPS = self.df.loc["Diluted EPS"].iloc[0]
            eps_row = self.df.loc["Diluted EPS"].dropna()
            n_years = len(eps_row) - 1
            
            # Calculate growth rate in percentage
            growth_rate = ((eps_row.iloc[0] / eps_row.iloc[-1]) ** (1 / n_years) - 1) * 100
            
            # Modern Graham Formula
            value = EPS * (8.5 + 2 * growth_rate) * (4.4 / bond_yield)
            self.results['graham_value'] = value
            return value
            
        except (KeyError, IndexError, ZeroDivisionError) as e:
            warnings.warn(f"Graham Value calculation failed: {e}")
            return None
    
    def calculate_all_valuations(self, **kwargs) -> Dict[str, float]:
        """
        Calculate all valuation models and return results.
        
        Returns:
            Dictionary containing all successful valuation results
        """
        self.calculate_dcf(**kwargs)
        self.calculate_ddm(**kwargs)
        self.calculate_pe_model()
        self.calculate_asset_based()
        self.calculate_graham_value()
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, float]:
        """Get all calculated valuation results."""
        return self.results.copy()
    
    def print_results(self):
        """Print all valuation results in a formatted way."""
        print("\n" + "="*50)
        print("INTRINSIC VALUATION RESULTS")
        print("="*50)
        
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
                print(f"{description}: ${round(value, 2)}")
        
        print("="*50)