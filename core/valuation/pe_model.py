from typing import Optional
import pandas as pd
from .base_valuation import BaseValuationModel
from core.config import MAX_REASONABLE_PE
from .confidence_scorer import ConfidenceScorer

class PEModel(BaseValuationModel):
    def calculate(self, use_median: bool = True) -> Optional[float]:
        model_warnings = []

        try:
            eps = self._get_eps()
            if eps is None or eps <= 0:
                model_warnings.append("Invalid EPS data")
                self.warnings = model_warnings
                return None

            if "P/E Ratio" not in self.df.index:
                model_warnings.append("P/E Ratio data not available")
                self.warnings = model_warnings
                return None

            pe_row = self.df.loc["P/E Ratio"].dropna()
            pe_filtered = pe_row[(pe_row > 0) & (pe_row < MAX_REASONABLE_PE)]

            if len(pe_filtered) == 0:
                model_warnings.append("No valid P/E ratios available")
                self.warnings = model_warnings
                return None

            if use_median:
                expected_pe = pe_filtered.median()
            else:
                expected_pe = pe_filtered.mean()

            value = eps * expected_pe

            # Use new confidence scorer
            confidence = ConfidenceScorer.score_pe(pe_filtered.values.tolist())

            self.confidence = confidence
            self.warnings = model_warnings
            return value

        except Exception as e:
            model_warnings.append(f"P/E Model calculation failed: {str(e)}")
            self.warnings = model_warnings
            return None

    def _get_eps(self) -> Optional[float]:
        if "Diluted EPS" in self.df.index:
            return self.df.loc["Diluted EPS"].iloc[0]
        elif "Basic EPS" in self.df.index:
            return self.df.loc["Basic EPS"].iloc[0]
        return None
