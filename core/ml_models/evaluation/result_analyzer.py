"""
Post-training analysis for ML models.
Includes feature importance and SHAP analysis.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

def get_feature_importances(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature_importances_")
    importances = model.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

def summarize_model_results(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """
    Returns basic summary of predictions vs. ground truth.
    """
    from .metrics import all_metrics
    metrics_summary = all_metrics(y_true, y_pred)
    return {'metrics': metrics_summary, 'y_true_mean': float(np.mean(y_true)), 'y_pred_mean': float(np.mean(y_pred))}
