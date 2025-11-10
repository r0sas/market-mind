"""
Sector-level analysis helpers for the combined dataframe.
"""

from typing import List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SectorAnalyzer:
    def __init__(self, combined_df: pd.DataFrame = None):
        self.combined_df = combined_df

    def attach(self, df: pd.DataFrame):
        self.combined_df = df

    def get_sector_summary(self, metric: str) -> pd.DataFrame:
        if self.combined_df is None:
            raise ValueError("No combined dataframe attached to SectorAnalyzer")
        if metric not in self.combined_df.columns:
            raise ValueError(f"Metric {metric} not found")
        sector_stats = self.combined_df.groupby('Sector')[metric].agg([
            ('count', 'count'), ('mean', 'mean'), ('median', 'median'),
            ('std', 'std'), ('min', 'min'), ('max', 'max')
        ]).round(2)
        return sector_stats

    def get_sector_data(self, sector: str) -> pd.DataFrame:
        if self.combined_df is None:
            raise ValueError("No combined dataframe attached to SectorAnalyzer")
        return self.combined_df[self.combined_df['Sector'] == sector].copy()

    def get_top_companies(self, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
        if self.combined_df is None:
            raise ValueError("No combined dataframe attached to SectorAnalyzer")
        if metric not in self.combined_df.columns:
            raise ValueError(f"Metric {metric} not found")
        return (self.combined_df.nsmallest(n, metric) if ascending else self.combined_df.nlargest(n, metric))

    def get_available_sectors(self) -> List[str]:
        if self.combined_df is None:
            raise ValueError("No combined dataframe attached to SectorAnalyzer")
        return sorted(self.combined_df['Sector'].unique().tolist())

    def get_available_metrics(self) -> List[str]:
        if self.combined_df is None:
            raise ValueError("No combined dataframe attached to SectorAnalyzer")
        exclude = ['Ticker', 'Company', 'Sector', 'Industry', 'Year', 'Market Cap']
        return [c for c in self.combined_df.columns if c not in exclude]
