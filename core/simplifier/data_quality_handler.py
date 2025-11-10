import pandas as pd
import logging
from typing import Tuple, List
from core.simplifier.constants import MIN_HISTORICAL_YEARS
from core.simplifier.simplifier_exceptions import SimplifierError

logger = logging.getLogger(__name__)

class DataQualityHandler:
    """Handles missing data and performs quality validation."""

    def __init__(self, prioritize_recent: bool = True):
        self.prioritize_recent = prioritize_recent

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans missing data according to the chosen strategy."""
        if df is None or df.empty:
            raise SimplifierError("Input DataFrame cannot be None or empty.")

        initial_cols = len(df.columns)

        if self.prioritize_recent:
            critical_metrics = ["Free Cash Flow", "Diluted EPS", "Basic EPS"]
            cols_to_keep = []

            for col in df.columns:
                col_data = df[col]
                has_critical_data = any(
                    pd.notna(col_data[m]) for m in critical_metrics if m in df.index
                )
                missing_pct = col_data.isna().sum() / len(col_data)

                if has_critical_data or missing_pct < 0.2:
                    cols_to_keep.append(col)

            df = df[cols_to_keep] if cols_to_keep else df.dropna(axis=1, how="any")
        else:
            df = df.dropna(axis=1, how="any")

        final_cols = len(df.columns)

        if final_cols == 0:
            raise SimplifierError("All columns removed due to missing data.")

        if final_cols < initial_cols:
            logger.info(f"Removed {initial_cols - final_cols} column(s) with missing data.")

        if final_cols < MIN_HISTORICAL_YEARS:
            logger.warning(f"Only {final_cols} years of data available (recommended ≥ {MIN_HISTORICAL_YEARS}).")

        return df

    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Perform data quality validation and return (is_valid, warnings)."""
        if df is None or df.empty:
            return False, ["Simplified DataFrame is None or empty"]

        warnings = []
        num_years = len(df.columns)

        if num_years < MIN_HISTORICAL_YEARS:
            warnings.append(f"Only {num_years} years of data (recommended ≥ {MIN_HISTORICAL_YEARS}).")

        critical_metrics = ["Free Cash Flow", "Diluted EPS", "Basic EPS", "Share Price"]
        missing_critical = [m for m in critical_metrics if m not in df.index]

        if missing_critical:
            warnings.append(f"Missing critical metrics: {', '.join(missing_critical)}")

        if "P/E Ratio" in df.index:
            pe = df.loc["P/E Ratio"].dropna()
            if len(pe) > 0:
                if (pe < 0).any():
                    warnings.append("Negative P/E ratios detected (negative earnings).")
                if (pe > 200).any():
                    warnings.append(f"Extremely high P/E ratios detected (max={pe.max():.1f}).")
                if pe.std() / (pe.mean() or 1) > 1.0:
                    warnings.append("High P/E volatility — valuation may be unreliable.")

        if "Free Cash Flow" in df.index:
            fcf = df.loc["Free Cash Flow"].dropna()
            if (fcf < 0).any():
                warnings.append(f"Negative Free Cash Flow in {int((fcf < 0).sum())} years.")

        if "Total Assets" in df.index and (df.loc["Total Assets"] <= 0).any():
            warnings.append("Non-positive Total Assets detected.")

        return len(missing_critical) == 0 and num_years >= 2, warnings
