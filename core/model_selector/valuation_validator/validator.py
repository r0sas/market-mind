import logging
from typing import Dict, Optional, Tuple
from .config import MIN_REASONABLE_PRICE, MAX_REASONABLE_PRICE, MAX_DEVIATION_FROM_CURRENT, MIN_DEVIATION_FROM_CURRENT
from .checks import check_currency_conversion_error, check_shares_error
from .model_specific import validate_asset_based

logger = logging.getLogger(__name__)

class ValuationValidator:
    def __init__(self, ticker: str, current_price: float):
        self.ticker = ticker
        self.current_price = current_price

    def validate_valuation(self, model_name: str, calculated_value: float, total_assets: Optional[float] = None, total_liabilities: Optional[float] = None, shares_outstanding: Optional[float] = None) -> Tuple[bool, Optional[float], list]:
        warnings = []
        corrected_value = calculated_value
        is_valid = True

        # Absolute bounds
        if calculated_value < MIN_REASONABLE_PRICE:
            warnings.append(f"{model_name}: Value ${calculated_value:.4f} too low")
            is_valid = False
        if calculated_value > MAX_REASONABLE_PRICE:
            warnings.append(f"{model_name}: Value ${calculated_value:.2f} too high")
            corrected_value, corrected = check_currency_conversion_error(self.ticker, calculated_value, self.current_price)
            if corrected:
                warnings.append(f"{model_name}: Corrected currency conversion to ${corrected_value:.2f}")
                is_valid = True

        # Deviation from current price
        if self.current_price and self.current_price > 0:
            ratio = calculated_value / self.current_price
            if ratio > MAX_DEVIATION_FROM_CURRENT and shares_outstanding:
                corrected_value, corrected = check_shares_error(calculated_value, total_assets, total_liabilities, shares_outstanding, MIN_REASONABLE_PRICE, MAX_REASONABLE_PRICE)
                if corrected:
                    warnings.append(f"{model_name}: Corrected shares outstanding, value now ${corrected_value:.2f}")
                    is_valid = True
                else:
                    is_valid = False
            elif ratio < MIN_DEVIATION_FROM_CURRENT:
                warnings.append(f"{model_name}: Value extremely low vs current price")
                is_valid = False

        # Model-specific
        if model_name.lower() in ['asset_based', 'asset-based', 'book value']:
            warnings.extend(validate_asset_based(calculated_value, total_assets, total_liabilities, shares_outstanding))

        return is_valid, corrected_value if is_valid else None, warnings
