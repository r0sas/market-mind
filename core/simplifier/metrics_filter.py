import pandas as pd
import logging
from typing import List
from core.simplifier.simplifier_exceptions import SimplifierError

logger = logging.getLogger(__name__)

class MetricsFilter:
    """Filters DataFrame to retain only essential metrics."""

    def __init__(self, df: pd.DataFrame, essential_metrics: List[str]):
        self.df = df.copy()
        self.essential_metrics = essential_metrics
        self.missing_metrics: List[str] = []

    def filter_essential_metrics(self) -> pd.DataFrame:
        """Keep only essential metrics present in the DataFrame."""
        existing_rows = [row for row in self.essential_metrics if row in self.df.index]
        self.missing_metrics = [row for row in self.essential_metrics if row not in self.df.index]

        if not existing_rows:
            raise SimplifierError(
                "No essential metrics found in the DataFrame. "
                f"Available metrics: {list(self.df.index[:10])}..."
            )

        if self.missing_metrics:
            logger.info(f"Missing metrics: {', '.join(self.missing_metrics)}")

        simplified_df = self.df.loc[existing_rows]
        logger.info(f"Filtered to {len(existing_rows)} essential metrics.")
        return simplified_df
