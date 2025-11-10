from typing import Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """
    Preprocess raw financial data for ML models.
    Tasks:
    - Handle missing values
    - Scale numerical features
    - Encode categorical features
    """

    def __init__(self, scaling_method: str = "standard"):
        """
        Args:
            scaling_method: "standard" for StandardScaler, "minmax" for MinMaxScaler
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()

    def fit_transform(self, df: pd.DataFrame, numeric_cols: Optional[list] = None) -> pd.DataFrame:
        """Fit scaler and transform numeric features"""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

        df_scaled = df.copy()
        df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
        return df_scaled

    def transform(self, df: pd.DataFrame, numeric_cols: Optional[list] = None) -> pd.DataFrame:
        """Transform numeric features using already fitted scaler"""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

        df_scaled = df.copy()
        df_scaled[numeric_cols] = self.scaler.transform(df_scaled[numeric_cols])
        return df_scaled
