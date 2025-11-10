"""
data_aggregation package

Provides a refactored MultiCompanyAggregator and helpers:
- company_fetcher: fetch & simplify a single company
- data_combiner: combine many company DataFrames into one dataset
- sector_analysis: sector-level utilities and summaries
- exporters: CSV / other export helpers
- display_tools: terminal-friendly summaries
- utils: small helpers (safe numeric conversion, delay)
"""

from .multi_company_aggregator import MultiCompanyAggregator

__all__ = ["MultiCompanyAggregator"]
