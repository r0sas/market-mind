"""
DataCombiner: combine multiple simplified company DataFrames into a single long dataframe.
Responsible for deduplication, column cleanup and safe numeric conversions.
"""

from typing import Dict, List, Optional
import pandas as pd
import logging
from .utils import safe_to_numeric

logger = logging.getLogger(__name__)

class DataCombiner:
    DEFAULT_NUMERIC_COLS = [
        'Market Cap', 'P/E Ratio', 'Basic EPS', 'Free Cash Flow',
        'Operating Cash Flow', 'Investing Cash Flow', 'Annual Dividends',
        'Net Income', 'Diluted EPS', 'Share Price', 'Total Assets',
        'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest',
        'Basic Average Shares', 'Diluted Average Shares', 'Year'
    ]

    def combine(
        self,
        company_data: Dict[str, pd.DataFrame],
        sector_info: Dict[str, dict],
        metrics: Optional[List[str]] = None,
        latest_year_only: bool = True
    ) -> pd.DataFrame:
        """
        Combine simplified company frames into a single DataFrame.

        Each simplified df is expected to have metrics on the index and years as columns
        (same structure as original code).
        """
        if not company_data:
            raise ValueError("No company data provided to DataCombiner.combine")

        combined_rows = []

        for ticker, df in company_data.items():
            if df is None:
                logger.debug(f"{ticker}: None dataframe, skipping")
                continue

            try:
                # remove duplicate rows/cols
                df = df.loc[~df.index.duplicated(keep='first')]
                df = df.loc[:, ~df.columns.duplicated(keep='first')]

                if df.empty:
                    logger.warning(f"{ticker}: DataFrame empty after dedupe, skipping")
                    continue

                info = sector_info.get(ticker, {})
                company_name = info.get("company_name", ticker)
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")
                market_cap = info.get("market_cap", None)

                if latest_year_only:
                    year_col = 'Year 1' if 'Year 1' in df.columns else df.columns[0]
                    row = {
                        "Ticker": ticker,
                        "Company": company_name,
                        "Sector": sector,
                        "Industry": industry,
                        "Market Cap": market_cap,
                        "Year": df.loc["Date", year_col] if "Date" in df.index else None
                    }
                    target_metrics = metrics if metrics else [idx for idx in df.index if idx not in ("Date", "Ticker")]
                    for m in target_metrics:
                        if m in df.index:
                            row[m] = df.loc[m, year_col]
                    combined_rows.append(row)
                else:
                    for col in df.columns:
                        year_val = df.loc["Date", col] if "Date" in df.index else col
                        row = {
                            "Ticker": ticker,
                            "Company": company_name,
                            "Sector": sector,
                            "Industry": industry,
                            "Market Cap": market_cap,
                            "Year": year_val
                        }
                        target_metrics = metrics if metrics else [idx for idx in df.index if idx not in ("Date", "Ticker")]
                        for m in target_metrics:
                            if m in df.index:
                                row[m] = df.loc[m, col]
                        combined_rows.append(row)
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")

        if not combined_rows:
            raise ValueError("No valid data rows after combining")

        combined_df = pd.DataFrame(combined_rows)

        # convert numeric columns safely
        safe_to_numeric(combined_df, self.DEFAULT_NUMERIC_COLS)

        return combined_df
