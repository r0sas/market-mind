from typing import List, Dict, Optional
import pandas as pd

class AggregatorInterface:
    """
    Interface for integrating MultiCompanyAggregator with ML pipelines.
    Provides methods to fetch, combine, and retrieve sector-specific data.
    """

    def fetch_data(self, tickers: List[str], use_concurrent: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch and simplify data for multiple tickers.
        Should return a dictionary mapping tickers to DataFrames.
        """
        raise NotImplementedError("Implement fetch_data using your MultiCompanyAggregator instance.")

    def get_combined_dataframe(self, metrics: Optional[List[str]] = None, latest_year_only: bool = True) -> pd.DataFrame:
        """
        Combine company data into a single DataFrame suitable for ML.
        """
        raise NotImplementedError("Implement get_combined_dataframe using aggregator data.")
