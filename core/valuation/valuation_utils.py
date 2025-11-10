# valuation/valuation_utils.py
"""
Utility functions for valuation models.
Provides helper functions for retrieving EPS, shares, and growth rates,
as well as general financial calculations shared across valuation models.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd


def get_eps(df: pd.DataFrame) -> Optional[float]:
    """Return the latest EPS value from the DataFrame."""
    if 'eps' in df.columns and not df['eps'].dropna().empty:
        return float(df['eps'].iloc[-1])
    elif 'net_income' in df.columns and 'shares_outstanding' in df.columns:
        eps = df['net_income'] / df['shares_outstanding']
        return float(eps.iloc[-1]) if not eps.dropna().empty else None
    return None


def get_shares_outstanding(df: pd.DataFrame) -> Optional[float]:
    """Retrieve the latest shares outstanding from DataFrame."""
    if 'shares_outstanding' in df.columns and not df['shares_outstanding'].dropna().empty:
        return float(df['shares_outstanding'].iloc[-1])
    return None


def get_growth_rate(df: pd.DataFrame, col: str = 'revenue') -> Optional[float]:
    """Estimate growth rate from a given financial column."""
    if col not in df.columns or len(df[col].dropna()) < 2:
        return None
    start, end = df[col].iloc[0], df[col].iloc[-1]
    if start <= 0:
        return None
    years = len(df[col].dropna()) - 1
    growth = (end / start) ** (1 / years) - 1
    return float(growth)


def validate_numeric(value: Optional[float]) -> Optional[float]:
    """Ensure a value is a valid positive float."""
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def apply_confidence_score(value: float, reference: float, thresholds: Dict[str, float]) -> str:
    """Assign a confidence score based on deviation from reference value."""
    if reference <= 0:
        return 'Unknown'
    deviation = abs(value - reference) / reference
    if deviation < thresholds['high']:
        return 'High'
    elif deviation < thresholds['medium']:
        return 'Medium'
    else:
        return 'Low'