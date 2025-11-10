"""
High-level MultiCompanyAggregator that orchestrates fetching, combining and analysis.

Public API mirrors the original class but uses the helper modules internally.
"""

from typing import List, Dict, Optional, Type
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .company_fetcher import CompanyFetcher
from .data_combiner import DataCombiner
from .sector_analysis import SectorAnalyzer
from .exporters import DataExporter
from .display_tools import print_summary

logger = logging.getLogger(__name__)

class MultiCompanyAggregator:
    def __init__(self, data_fetcher_class: Type, simplifier_class: Type, max_workers: int = 5, delay: float = 0.5):
        """
        Args:
            data_fetcher_class: class for fetching raw financials by ticker
            simplifier_class: class for simplifying those raw financials
            max_workers: thread pool size for concurrent fetches
            delay: small delay after each fetch to reduce rate-limits
        """
        self.data_fetcher_class = data_fetcher_class
        self.simplifier_class = simplifier_class
        self.max_workers = max_workers
        self.delay = delay

        self.company_data: Dict[str, pd.DataFrame] = {}
        self.sector_info: Dict[str, dict] = {}
        self.combined_df: Optional[pd.DataFrame] = None

        # helper components
        self.fetcher = CompanyFetcher(self.data_fetcher_class, self.simplifier_class, delay=self.delay)
        self.combiner = DataCombiner()
        self.analyzer = SectorAnalyzer()
        self.exporter = DataExporter()

    def fetch_company_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Compatibility wrapper (single-company fetch)."""
        _, df, summary = self.fetcher.fetch(ticker)
        if df is not None:
            self.company_data[ticker] = df
            self.sector_info[ticker] = {
                'company_name': (summary or {}).get('company_name', 'N/A'),
                'sector': (summary or {}).get('sector', 'N/A'),
                'industry': (summary or {}).get('industry', 'N/A'),
                'market_cap': (summary or {}).get('market_cap', None)
            }
        return df

    def fetch_multiple_companies(self, tickers: List[str], use_concurrent: bool = True) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        if use_concurrent:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {executor.submit(self.fetcher.fetch, t): t for t in tickers}
                for fut in as_completed(future_map):
                    t = future_map[fut]
                    try:
                        ticker, df, summary = fut.result()
                        if df is not None:
                            results[ticker] = df
                            self.company_data[ticker] = df
                            self.sector_info[ticker] = {
                                'company_name': (summary or {}).get('company_name', 'N/A'),
                                'sector': (summary or {}).get('sector', 'N/A'),
                                'industry': (summary or {}).get('industry', 'N/A'),
                                'market_cap': (summary or {}).get('market_cap', None)
                            }
                        else:
                            logger.warning(f"No data returned for {t}")
                    except Exception as e:
                        logger.warning(f"Exception for {t}: {e}")
        else:
            for t in tickers:
                logger.info(f"Fetching data for {t}")
                _, df, summary = self.fetcher.fetch(t)
                if df is not None:
                    results[t] = df
                    self.company_data[t] = df
                    self.sector_info[t] = {
                        'company_name': (summary or {}).get('company_name', 'N/A'),
                        'sector': (summary or {}).get('sector', 'N/A'),
                        'industry': (summary or {}).get('industry', 'N/A'),
                        'market_cap': (summary or {}).get('market_cap', None)
                    }
                else:
                    logger.warning(f"No data for {t}")

        return results

    def create_combined_dataframe(self, metrics: Optional[List[str]] = None, latest_year_only: bool = True) -> pd.DataFrame:
        """
        Wraps DataCombiner.combine and attaches resulting df to SectorAnalyzer.
        """
        self.combined_df = self.combiner.combine(self.company_data, self.sector_info, metrics=metrics, latest_year_only=latest_year_only)
        self.analyzer.attach(self.combined_df)
        return self.combined_df

    # Sector analysis proxies
    def get_sector_summary(self, metric: str):
        return self.analyzer.get_sector_summary(metric)

    def get_sector_data(self, sector: str):
        return self.analyzer.get_sector_data(sector)

    def get_top_companies(self, metric: str, n: int = 10, ascending: bool = False):
        return self.analyzer.get_top_companies(metric, n=n, ascending=ascending)

    # Export / display
    def export_to_csv(self, filename: str):
        if self.combined_df is None:
            raise ValueError("No combined data available")
        return self.exporter.to_csv(self.combined_df, filename)

    def display_summary(self):
        print_summary(len(self.company_data), self.sector_info, self.combined_df)

    def get_available_sectors(self):
        return self.analyzer.get_available_sectors()

    def get_available_metrics(self):
        return self.analyzer.get_available_metrics()
