from typing import Optional
import pandas as pd

def get_eps(df: pd.DataFrame) -> Optional[float]:
    """
    Get the most recent EPS value, preferring Diluted EPS over Basic EPS.

    Args:
        df: Simplified financial DataFrame

    Returns:
        EPS value or None if unavailable
    """
    if "Diluted EPS" in df.index:
        return df.loc["Diluted EPS"].iloc[0]
    elif "Basic EPS" in df.index:
        return df.loc["Basic EPS"].iloc[0]
    return None

def get_shares_outstanding(df: pd.DataFrame) -> Optional[float]:
    """
    Get the number of shares outstanding, preferring Basic over Diluted.

    Args:
        df: Simplified financial DataFrame

    Returns:
        Number of shares or None if unavailable
    """
    if "Basic Average Shares" in df.index:
        return df.loc["Basic Average Shares"].iloc[0]
    elif "Diluted Average Shares" in df.index:
        return df.loc["Diluted Average Shares"].iloc[0]
    return None
