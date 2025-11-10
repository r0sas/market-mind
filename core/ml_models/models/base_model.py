from typing import Any, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

class BaseModel:
    """
    Base class for all ML models.
    Provides interface for training, prediction, and evaluation.
    """

    def __init__(self, model: BaseEstimator = None, name: str = "BaseModel"):
        self.model = model
        self.name = name
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to data"""
        if self.model is None:
            raise NotImplementedError("Model instance must be provided")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict target values"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        preds = self.predict(X)
        # Placeholder for metrics
        return {'dummy_metric': np.mean(preds - y)}
