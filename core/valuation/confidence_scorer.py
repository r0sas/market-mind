# valuation/confidence_scorer.py
from typing import Optional
import pandas as pd
import numpy as np
from core.config import CONFIDENCE_THRESHOLDS

class ConfidenceScorer:
    """
    Calculates confidence scores for valuation models.
    Uses coefficient of variation (CV) or other metrics to determine High/Medium/Low confidence.
    """

    @staticmethod
    def score_fcf(fcf_series: pd.Series) -> str:
        """
        Confidence based on Free Cash Flow volatility.
        """
        if fcf_series is None or len(fcf_series) == 0:
            return 'Low'

        fcf_cv = fcf_series.std() / fcf_series.mean()
        if fcf_cv > CONFIDENCE_THRESHOLDS.get('fcf_volatility_high', 0.3):
            return 'Low'
        elif fcf_cv > CONFIDENCE_THRESHOLDS.get('fcf_volatility_medium', 0.15):
            return 'Medium'
        else:
            return 'High'

    @staticmethod
    def score_pe(pe_series: pd.Series) -> str:
        """
        Confidence based on P/E ratio volatility.
        """
        if pe_series is None or len(pe_series) == 0:
            return 'Low'

        pe_series = pe_series[pe_series > 0]  # Only positive values
        if len(pe_series) == 0:
            return 'Low'

        pe_cv = pe_series.std() / pe_series.mean()
        if pe_cv > CONFIDENCE_THRESHOLDS.get('pe_volatility_high', 0.5):
            return 'Low'
        elif pe_cv > CONFIDENCE_THRESHOLDS.get('pe_volatility_medium', 0.3):
            return 'Medium'
        else:
            return 'High'

    @staticmethod
    def score_generic(value_series: pd.Series, high_thresh: float = 0.3, med_thresh: float = 0.15) -> str:
        """
        Generic confidence scorer using coefficient of variation.
        """
        if value_series is None or len(value_series) == 0:
            return 'Low'

        cv = value_series.std() / value_series.mean()
        if cv > high_thresh:
            return 'Low'
        elif cv > med_thresh:
            return 'Medium'
        else:
            return 'High'

    @staticmethod
    def from_ratio(volatility: float, high_thresh: float = 0.5, med_thresh: float = 0.3) -> str:
        """
        Confidence directly from a coefficient of variation or ratio.
        """
        if volatility > high_thresh:
            return 'Low'
        elif volatility > med_thresh:
            return 'Medium'
        else:
            return 'High'
