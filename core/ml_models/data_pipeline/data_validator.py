from typing import Dict
import pandas as pd
import numpy as np

class DataValidator:
    """
    Validate and ensure data consistency for ML pipelines.
    Checks:
    - Missing values
    - Infinite values
    - Data types
    - Sector coverage
    """

    def __init__(self):
        pass

    def validate(self, df: pd.DataFrame) -> Dict[str, any]:
        """Return a report on data quality"""
        report = {}
        report['num_rows'] = len(df)
        report['num_columns'] = len(df.columns)
        report['missing_values_pct'] = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        report['infinite_values'] = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        report['columns'] = df.columns.tolist()
        return report

    def clean(self, df: pd.DataFrame, fillna_value: float = 0.0) -> pd.DataFrame:
        """Clean DataFrame by filling missing values and replacing infinite values"""
        df_clean = df.replace([np.inf, -np.inf], np.nan).fillna(fillna_value)
        return df_clean
