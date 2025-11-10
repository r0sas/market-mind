"""
CompanyFetcher: encapsulates fetching and simplifying data for a single company.
Used by MultiCompanyAggregator to parallelize requests and isolate fetch logic.
"""

from typing import Optional, Type, Tuple, Dict
import logging
import warnings
import time
import pandas as pd
from .utils import sleep_delay

logger = logging.getLogger(__name__)

class CompanyFetcher:
    def __init__(self, data_fetcher_class: Type, simplifier_class: Type, delay: float = 0.5):
        """
        Args:
            data_fetcher_class: class (callable) that when instantiated with ticker provides:
                - get_comprehensive_data() -> pd.DataFrame
                - get_summary() -> dict
            simplifier_class: class that when instantiated with comprehensive_df provides:
                - simplify() -> pd.DataFrame
            delay: seconds to wait after a fetch (rate limiting)
        """
        self.data_fetcher_class = data_fetcher_class
        self.simplifier_class = simplifier_class
        self.delay = delay

    def fetch(self, ticker: str) -> Tuple[str, Optional[pd.DataFrame], Optional[dict]]:
        """
        Fetch and simplify a single ticker.

        Returns:
            (ticker, simplified_df or None, summary dict or None)
        """
        try:
            fetcher = self.data_fetcher_class(ticker)
            comprehensive_df = fetcher.get_comprehensive_data()
            if comprehensive_df is None or (hasattr(comprehensive_df, "empty") and comprehensive_df.empty):
                warnings.warn(f"Comprehensive data for {ticker} is empty.")
                return ticker, None, None

            summary = fetcher.get_summary() or {}
            simplifier = self.simplifier_class(comprehensive_df)
            # support both simplify() and simplify_for_ml()
            if hasattr(simplifier, "simplify_for_ml"):
                simplified_df = simplifier.simplify_for_ml()
            else:
                simplified_df = simplifier.simplify()

            if simplified_df is None or (hasattr(simplified_df, "empty") and simplified_df.empty):
                warnings.warn(f"Simplified data for {ticker} is empty.")
                return ticker, None, summary

            # Delay to respect rate limits
            sleep_delay(self.delay)
            return ticker, simplified_df, summary

        except Exception as e:
            warnings.warn(f"Failed to fetch/simplify data for {ticker}: {e}")
            logger.exception(e)
            # still sleep on failure to avoid hammering endpoints
            sleep_delay(self.delay)
            return ticker, None, None
