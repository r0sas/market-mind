"""
Small helper utilities used by the data_aggregation package.
"""

import time
import logging
from typing import Iterable, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def safe_to_numeric(df: pd.DataFrame, cols: Iterable[str]):
    """Convert columns in-place to numeric, coercing errors to NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def ensure_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def sleep_delay(seconds: float):
    """Simple wrapper for time.sleep so callers are explicit."""
    if seconds and seconds > 0:
        time.sleep(seconds)
