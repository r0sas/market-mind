from typing import Optional
import logging
from core.valuation.base_valuation import BaseValuationModel
from core.valuation.utils import get_eps

logger = logging.getLogger(__name__)

DEFAULT_BOND_YIELD = 0.0521  # Can be overridden from Config if needed

class GrahamModel(BaseValuationModel):
    """Modern Graham Formula Valuation Model"""

    def calculate(self, bond_yield: float = DEFAULT_BOND_YIELD) -> Optional[float]:
        model_warnings = []

        try:
            eps = get_eps(self.df)
            if eps is None or eps <= 0:
                model_warnings.append("Invalid EPS data for Graham formula")
                self.warnings = model_warnings
                return None

            if "Diluted EPS" not in self.df.index:
                model_warnings.append("EPS data not available for growth calculation")
                self.warnings = model_warnings
                return None

            eps_row = self.df.loc["Diluted EPS"].dropna()
            if len(eps_row) < 2:
                model_warnings.append("Insufficient EPS history for growth calculation")
                self.warnings = model_warnings
                return None

            eps_latest = eps_row.iloc[0]
            eps_earliest = eps_row.iloc[-1]
            n_years = len(eps_row) - 1

            if eps_earliest <= 0:
                model_warnings.append("Cannot calculate growth: negative or zero historical EPS")
                self.warnings = model_warnings
                return None

            # Growth rate in decimal
            growth_rate_decimal = (eps_latest / eps_earliest) ** (1 / n_years) - 1
            growth_rate_pct = growth_rate_decimal * 100

            # Cap growth rate at 20% per Graham suggestion
            if growth_rate_pct > 20:
                model_warnings.append(f"Growth rate ({growth_rate_pct:.1f}%) capped at 20% for Graham formula")
                growth_rate_pct = 20

            value = eps * (8.5 + 2 * growth_rate_pct) * (4.4 / (bond_yield * 100))

            self.results = {'graham_value': value}
            self.confidence = 'Medium'
            self.warnings = model_warnings

            logger.info(f"Graham Value: ${value:.2f} (Growth: {growth_rate_pct:.1f}%)")
            return value

        except Exception as e:
            model_warnings.append(f"Graham Value calculation failed: {str(e)}")
            self.warnings = model_warnings
            return None
