"""
Cross-validation and hyperparameter tuning utilities.
"""

from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
    """
    Perform k-fold cross-validation on a model.

    Args:
        model: Fitted or unfitted model
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric ('r2', 'neg_mean_squared_error', etc.)

    Returns:
        Dictionary with mean and std of scores
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return {'mean_score': float(scores.mean()), 'std_score': float(scores.std())}
