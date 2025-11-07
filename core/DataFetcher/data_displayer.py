import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataDisplayer:
    """Stateless utility class for displaying financial data."""
    
    @staticmethod
    def display_data_overview(
        ticker_symbol: str,
        summary: Dict[str, Any],
        combined_df: pd.DataFrame
    ) -> None:
        """
        Display an overview of fetched data.
        
        Args:
            ticker_symbol: Stock ticker symbol
            summary: Summary dictionary with company info
            combined_df: The combined DataFrame to summarize
        """
        print(f"\n{'='*50}")
        print(f"DATA OVERVIEW FOR {ticker_symbol}")
        print(f"{'='*50}")
        
        print(f"Company: {summary['company_name']}")
        print(f"Sector: {summary['sector']}")
        print(f"Industry: {summary['industry']}")
        
        if summary['market_cap']:
            print(f"Market Cap: ${summary['market_cap']:,.0f}")
        if summary['current_price']:
            print(f"Current Price: ${summary['current_price']:.2f}")
        if summary['pe_ratio']:
            print(f"P/E Ratio: {summary['pe_ratio']:.2f}")
        
        print(f"\nCombined DataFrame shape: {combined_df.shape}")
        print(f"Time periods covered: {len(combined_df.columns)}")