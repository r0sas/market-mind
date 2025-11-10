"""
Terminal/CLI display helpers for aggregated data.
"""

from typing import Dict
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def print_summary(company_count: int, sector_info: Dict[str, Dict], combined_df: pd.DataFrame = None):
    print("\n" + "="*60)
    print("MULTI-COMPANY DATA SUMMARY")
    print("="*60)
    print(f"Total companies: {company_count}")
    print(f"\nCompanies by ticker:")
    for ticker, info in sector_info.items():
        market_cap = info.get('market_cap')
        mc_str = f"${market_cap:,.0f}" if market_cap else "N/A"
        print(f"  {ticker}: {info.get('company_name','N/A')} ({info.get('sector','N/A')}) - Market Cap: {mc_str}")

    if combined_df is not None:
        print(f"\nCombined DataFrame shape: {combined_df.shape}")
        sectors = sorted(combined_df['Sector'].unique().tolist()) if 'Sector' in combined_df.columns else []
        print(f"Sectors represented: {len(sectors)}")
        print(f"Sectors: {', '.join(sectors)}")
        metrics = [c for c in combined_df.columns if c not in ('Ticker','Company','Sector','Industry','Year','Market Cap')]
        print(f"\nAvailable metrics ({len(metrics)}):")
        for m in metrics:
            print(f"  - {m}")

    print("="*60)
