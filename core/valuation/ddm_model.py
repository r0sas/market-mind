from typing import Optional
from .base_valuation import BaseValuationModel
from core.config import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_PROJECTION_YEARS,
    MAX_REASONABLE_GROWTH_RATE
)
from .confidence_scorer import ConfidenceScorer

class DDMModel(BaseValuationModel):
    def calculate(
        self,
        required_rate: float = DEFAULT_DISCOUNT_RATE,
        terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
        projection_years: int = DEFAULT_PROJECTION_YEARS
    ) -> Optional[float]:
        model_warnings = []

        try:
            if required_rate <= terminal_growth:
                model_warnings.append("Required rate must be greater than terminal growth")
                self.warnings = model_warnings
                return None

            if "Annual Dividends" not in self.df.index:
                model_warnings.append("Dividend data not available")
                self.warnings = model_warnings
                return None

            dividends_row = self.df.loc["Annual Dividends"]
            if dividends_row.sum() == 0:
                model_warnings.append("Company does not pay dividends")
                self.warnings = model_warnings
                return None

            div_latest = dividends_row.iloc[0]
            div_earliest = dividends_row.iloc[-1]
            n_years = len(dividends_row) - 1

            if div_earliest <= 0 or div_latest <= 0:
                model_warnings.append("Cannot calculate DDM: negative or zero dividends")
                self.warnings = model_warnings
                return None

            growth_rate = (div_latest / div_earliest) ** (1 / n_years) - 1
            if growth_rate > MAX_REASONABLE_GROWTH_RATE:
                model_warnings.append(f"Warning: Dividend growth rate ({growth_rate*100:.1f}%) is very high")

            # Multi-stage DDM
            pv_dividends = sum(
                div_latest * (1 + growth_rate) ** t / (1 + required_rate) ** t
                for t in range(1, projection_years + 1)
            )

            terminal_dividend = div_latest * (1 + growth_rate) ** projection_years * (1 + terminal_growth)
            pv_terminal = terminal_dividend / (required_rate - terminal_growth) / (1 + required_rate) ** projection_years

            value = pv_dividends + pv_terminal

            # Use confidence scorer (reusing FCF logic for simplicity, could create dividend-specific)
            confidence = ConfidenceScorer.score_fcf(dividends_row.values.tolist())

            self.confidence = confidence
            self.warnings = model_warnings
            return value

        except Exception as e:
            model_warnings.append(f"DDM calculation failed: {str(e)}")
            self.warnings = model_warnings
            return None
