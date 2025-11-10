"""
Export helpers: CSV, Excel, JSON as needed.
"""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataExporter:
    def to_csv(self, df: pd.DataFrame, filename: str, index: bool = False):
        df.to_csv(filename, index=index)
        logger.info(f"Exported dataframe to CSV: {filename}")
        return filename

    def to_excel(self, df: pd.DataFrame, filename: str, sheet_name: str = "data"):
        df.to_excel(filename, sheet_name=sheet_name, index=False)
        logger.info(f"Exported dataframe to Excel: {filename}")
        return filename

    def to_json(self, df: pd.DataFrame, filename: str, orient: str = "records"):
        df.to_json(filename, orient=orient, date_format="iso")
        logger.info(f"Exported dataframe to JSON: {filename}")
        return filename
