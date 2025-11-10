from typing import Optional
import logging
from core.valuation.base_valuation import BaseValuationModel
from core.valuation.utils import get_shares_outstanding

logger = logging.getLogger(__name__)


class AssetBasedModel(BaseValuationModel):
    """Asset-Based (Book Value) Valuation Model"""

    def calculate(self) -> Optional[float]:
        model_warnings = []

        try:
            if "Total Assets" not in self.df.index:
                model_warnings.append("Total Assets data not available")
                self.warnings = model_warnings
                return None

            if "Total Liabilities Net Minority Interest" not in self.df.index:
                model_warnings.append("Total Liabilities data not available")
                self.warnings = model_warnings
                return None

            total_assets = self.df.loc["Total Assets"].iloc[0]
            total_liabilities = self.df.loc["Total Liabilities Net Minority Interest"].iloc[0]
            shares_outstanding = get_shares_outstanding(self.df)

            if shares_outstanding is None or shares_outstanding <= 0:
                model_warnings.append("Invalid shares outstanding data")
                self.warnings = model_warnings
                return None

            book_value_per_share = (total_assets - total_liabilities) / shares_outstanding
            if book_value_per_share <= 0:
                model_warnings.append("Book value is negative (liabilities exceed assets)")

            self.results = {'asset_based': book_value_per_share}
            self.confidence = 'Medium'
            self.warnings = model_warnings

            logger.info(f"Asset-Based Value: ${book_value_per_share:.2f}")
            return book_value_per_share

        except Exception as e:
            model_warnings.append(f"Asset-Based valuation failed: {str(e)}")
            self.warnings = model_warnings
            return None
