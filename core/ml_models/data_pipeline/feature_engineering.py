from typing import Optional
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Generate ML-ready features from financial data.
    Examples:
    - Growth rates
    - Ratios (profitability, leverage, liquidity)
    - Composite scores
    """

    def __init__(self):
        pass

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add commonly used derived features for ML models"""
        df_new = df.copy()

        # Example: EPS growth YoY
        if 'Basic EPS' in df.index and df.shape[1] > 1:
            eps_values = df.loc['Basic EPS'].values
            eps_growth = [(eps_values[i] / eps_values[i+1] - 1) * 100 
                          if eps_values[i+1] != 0 else np.nan 
                          for i in range(len(eps_values)-1)]
            eps_growth.append(np.nan)
            df_new.loc['EPS Growth YoY'] = eps_growth

        # Example: Profit margin
        if 'Net Income' in df.index and 'Total Revenue' in df.index:
            df_new.loc['Profit Margin'] = (df.loc['Net Income'] / df.loc['Total Revenue']) * 100

        # Example: Current ratio
        if 'Current Assets' in df.index and 'Current Liabilities' in df.index:
            df_new.loc['Current Ratio'] = df.loc['Current Assets'] / df.loc['Current Liabilities']

        return df_new
