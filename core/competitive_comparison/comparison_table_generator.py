"""
comparison_table_generator.py

Turn fetched data into a human-friendly pandas DataFrame 'league table'.
"""

from typing import Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ComparisonTableGenerator:
    """
    Builds a comparison DataFrame from the fetched data dictionary.
    """

    def __init__(self, data: Dict[str, Dict]):
        self.data = data

    def build_table(self) -> pd.DataFrame:
        rows = []
        for ticker, d in (self.data or {}).items():
            if not d:
                continue
            rows.append({
                'Company': d.get('company_name', ticker),
                'Ticker': ticker,
                'Price': f"${d['current_price']:.2f}" if d.get('current_price') else "N/A",
                'Market Cap': f"${d['market_cap']/1e9:.2f}B" if d.get('market_cap') else "N/A",
                'P/E Ratio': f"{d['pe_ratio']:.2f}" if d.get('pe_ratio') else "N/A",
                'Div Yield': f"{d['dividend_yield']*100:.2f}%" if d.get('dividend_yield') else "0%",
                '3M Return': f"{d['return_3m']:+.1f}%" if d.get('return_3m') is not None else "N/A",
                '6M Return': f"{d['return_6m']:+.1f}%" if d.get('return_6m') is not None else "N/A",
                '1Y Return': f"{d['return_1y']:+.1f}%" if d.get('return_1y') is not None else "N/A",
                'Profit Margin': f"{d['profit_margin']*100:.1f}%" if d.get('profit_margin') else "N/A",
                'ROE': f"{d['roe']*100:.1f}%" if d.get('roe') else "N/A",
                'Beta': f"{d['beta']:.2f}" if d.get('beta') else "N/A",
            })

        df = pd.DataFrame(rows)
        # Optionally reorder columns
        cols = ['Company', 'Ticker', 'Price', 'Market Cap', 'P/E Ratio', 'Div Yield',
                '3M Return', '6M Return', '1Y Return', 'Profit Margin', 'ROE', 'Beta']
        df = df[cols] if not df.empty else df
        logger.info("Comparison table generated")
        return df
