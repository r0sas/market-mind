import pandas as pd
import logging
from typing import List, Optional, Dict
from core.simplifier.constants import ESSENTIAL_METRICS
from core.simplifier.simplifier_exceptions import SimplifierError
from core.simplifier.column_transformer import ColumnTransformer
from core.simplifier.metrics_filter import MetricsFilter
from core.simplifier.data_quality_handler import DataQualityHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IVSimplifier:
    """Main orchestrator for financial data simplification."""

    def __init__(self, comprehensive_df: pd.DataFrame,
                 essential_metrics: Optional[List[str]] = None,
                 prioritize_recent: bool = True):
        if comprehensive_df is None or comprehensive_df.empty:
            raise SimplifierError("comprehensive_df cannot be None or empty")

        self.comprehensive_df = comprehensive_df.copy()
        self.simplified_df: Optional[pd.DataFrame] = None
        self.essential_metrics = essential_metrics or ESSENTIAL_METRICS
        self.prioritize_recent = prioritize_recent
        self.data_quality_warnings: List[str] = []
        self.handler = DataQualityHandler(prioritize_recent)

    def simplify(self) -> pd.DataFrame:
        """Execute the full simplification pipeline."""
        try:
            df = ColumnTransformer(self.comprehensive_df).transform_columns_to_years()
            df = MetricsFilter(df, self.essential_metrics).filter_essential_metrics()
            df = self.handler.handle_missing_data(df)

            self.simplified_df = df
            is_valid, warnings = self.handler.validate_data_quality(df)
            self.data_quality_warnings = warnings

            if warnings:
                logger.warning("Data quality issues detected:")
                for w in warnings:
                    logger.warning(f" - {w}")

            if not is_valid:
                logger.error("Data validation failed. Results may be unreliable.")

            return df

        except Exception as e:
            raise SimplifierError(f"Simplification failed: {str(e)}")

    def get_simplified_data(self) -> pd.DataFrame:
        """Return the simplified DataFrame."""
        if self.simplified_df is None:
            raise SimplifierError("No simplified data available. Call simplify() first.")
        return self.simplified_df.copy()

    def get_original_data(self) -> pd.DataFrame:
        """Return the original DataFrame (after column transformation)."""
        return self.comprehensive_df.copy()

    def get_data_quality_report(self) -> Dict[str, any]:
        """Return structured data quality diagnostics."""
        if self.simplified_df is None:
            return {"error": "No simplified data available. Call simplify() first."}

        num_years = len(self.simplified_df.columns)
        report = {
            "num_years": num_years,
            "num_metrics": len(self.simplified_df.index),
            "warnings": self.data_quality_warnings,
        }

        if "Date" in self.simplified_df.index:
            years = self.simplified_df.loc["Date"].dropna().values
            report["year_range"] = f"{int(min(years))} - {int(max(years))}"

        return report

    def display_summary(self) -> None:
        """Display a formatted summary of simplified data."""
        if self.simplified_df is None:
            print("No simplified data available. Call simplify() first.")
            return

        print("\n" + "=" * 50)
        print("SIMPLIFIED DATA SUMMARY")
        print("=" * 50)
        print(f"Metrics included: {len(self.simplified_df)}")
        print(f"Years of data: {len(self.simplified_df.columns)}")

        if "Date" in self.simplified_df.index:
            years = self.simplified_df.loc["Date"].dropna().values
            print(f"Year range: {int(min(years))} - {int(max(years))}")

        print("\nMetrics available:")
        for metric in self.simplified_df.index:
            if metric not in ["Date", "Ticker"]:
                print(f" - {metric}")

        if self.data_quality_warnings:
            print("\n" + "=" * 50)
            print("DATA QUALITY WARNINGS")
            print("=" * 50)
            for warning in self.data_quality_warnings:
                print(f"⚠️ {warning}")
