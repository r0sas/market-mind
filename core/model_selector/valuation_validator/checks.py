from typing import Tuple, Optional
from .config import FOREIGN_TICKERS, CURRENCY_CONVERSION

def check_currency_conversion_error(ticker: str, value: float, current_price: float) -> Tuple[float, bool]:
    if ticker in FOREIGN_TICKERS:
        currency = FOREIGN_TICKERS[ticker]
        rate = CURRENCY_CONVERSION.get(currency)
        if rate and value / rate < current_price * 2:
            return value / rate, True
    # Additional heuristics (Yen detection)
    if value > 10000 and value < 100000:
        corrected = value / 150
        if abs(corrected - current_price) / current_price < 0.5:
            return corrected, True
    return value, False

def check_shares_error(value: float, total_assets: float, total_liabilities: float, shares_outstanding: float, min_price: float, max_price: float) -> Tuple[float, bool]:
    book_value = total_assets - total_liabilities
    if value > 1000:
        for factor in [1000, 1_000_000]:
            corrected_value = book_value / (shares_outstanding * factor)
            if min_price < corrected_value < max_price:
                return corrected_value, True
    return value, False
