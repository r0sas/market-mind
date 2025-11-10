from typing import Optional
import numpy as np
from .base_valuation import BaseValuationModel
from core.config import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_PROJECTION_YEARS,
    MAX_REASONABLE_GROWTH_RATE
)
from .confidence_scorer import ConfidenceScorer

class DCFModel(BaseValuationModel):
    def calculate(
        self,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
        terminal_growth_rate: float = DEFAULT_TERMINAL_GROWTH,
        projection_years: int = DEFAULT_PROJECTION_YEARS,
        custom_growth_rate: Optional[float] = None
    ) -> Optional[float]:
        model_warnings = []
        try:
            if discount_rate <= terminal_growth_rate:
                model_warnings.append("Discount rate must be greater than terminal growth rate")
                self.warnings = model_warnings
                return None

            if "Free Cash Flow" not in self.df.index:
                model_warnings.append("Free Cash Flow data not available")
                self.warnings = model_warnings
                return None

            fcf_row = self.df.loc["Free Cash Flow"]
            fcf_latest = fcf_row.iloc[0]
            fcf_earliest = fcf_row.iloc[-1]
            n_years = len(fcf_row) - 1

            if fcf_earliest <= 0 or fcf_latest <= 0:
                model_warnings.append("Cannot calculate DCF: negative or zero FCF")
                self.warnings = model_warnings
                return None

            cagr = custom_growth_rate if custom_growth_rate is not None else (fcf_latest / fcf_earliest) ** (1 / n_years) - 1

            if cagr > MAX_REASONABLE_GROWTH_RATE:
                model_warnings.append(f"Warning: Growth rate ({cagr*100:.1f}%) is very high")

            # Project future FCFs
            present_value_fcf = [
                fcf_latest * (1 + cagr) ** year / (1 + discount_rate) ** year
                for year in range(1, projection_years + 1)
            ]

            terminal_value = (fcf_latest * (1 + cagr) ** projection_years * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
            pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

            enterprise_value = sum(present_value_fcf) + pv_terminal
            shares_outstanding = self._get_shares_outstanding()

            if shares_outstanding is None or shares_outstanding <= 0:
                model_warnings.append("Invalid shares outstanding data")
                self.warnings = model_warnings
                return None

            value_per_share = enterprise_value / shares_outstanding

            # Use the new confidence scorer
            confidence = ConfidenceScorer.score_fcf(fcf_row.values.tolist())

            self.confidence = confidence
            self.warnings = model_warnings
            return value_per_share

        except Exception as e:
            model_warnings.append(f"DCF calculation failed: {str(e)}")
            self.warnings = model_warnings
            return None

    def _get_shares_outstanding(self) -> Optional[float]:
        if "Basic Average Shares" in self.df.index:
            return self.df.loc["Basic Average Shares"].iloc[0]
        elif "Diluted Average Shares" in self.df.index:
            return self.df.loc["Diluted Average Shares"].iloc[0]
        return None
