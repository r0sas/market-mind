import pandas as pd
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Stateless utility class for calculating financial metrics."""
    
    @staticmethod
    def calculate_pe_ratio(
        price: Optional[float], 
        earnings_per_share: Optional[float]
    ) -> Optional[float]:
        """
        Calculate P/E ratio.
        
        Args:
            price: Share price
            earnings_per_share: EPS value
            
        Returns:
            P/E ratio or None if calculation not possible
        """
        if not price or not earnings_per_share or earnings_per_share == 0:
            return None
        return price / earnings_per_share
    
    @staticmethod
    def calculate_pe_ratios(
        enhanced_is: pd.DataFrame,
        eps_columns: List[str]
    ) -> List[Optional[float]]:
        """
        Calculate P/E ratios for each period.
        
        Args:
            enhanced_is: Enhanced income statement with share prices
            eps_columns: List of EPS column names to try
            
        Returns:
            List of P/E ratios
        """
        pe_ratios = []
        for col in enhanced_is.columns:
            share_price = (
                enhanced_is.loc["Share Price", col] 
                if "Share Price" in enhanced_is.index 
                else None
            )
            
            # Try to find EPS from available columns
            eps = None
            for eps_name in eps_columns:
                if eps_name in enhanced_is.index:
                    eps = enhanced_is.loc[eps_name, col]
                    break
            
            pe = MetricsCalculator.calculate_pe_ratio(share_price, eps)
            pe_ratios.append(pe)
        
        return pe_ratios