"""
Common regression evaluation metrics.
"""

from typing import Dict
import numpy as np
import pandas as pd

def mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return np.nan
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)

def all_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
