from typing import List, Optional

def validate_asset_based(calculated_value: float, total_assets: Optional[float], total_liabilities: Optional[float], shares_outstanding: Optional[float]) -> List[str]:
    warnings = []
    if not all([total_assets, total_liabilities, shares_outstanding]):
        warnings.append("Asset-based: Missing balance sheet data")
        return warnings

    book_value = total_assets - total_liabilities
    expected_per_share = book_value / shares_outstanding

    if abs(calculated_value - expected_per_share) / expected_per_share > 0.01:
        warnings.append(f"Asset-based: Calculated ${calculated_value:.2f} differs from expected ${expected_per_share:.2f}")

    if book_value < 0:
        warnings.append("Asset-based: Negative book value (liabilities > assets)")

    if shares_outstanding < 1000:
        warnings.append("Asset-based: Shares outstanding seems too low - possible wrong units")
    if shares_outstanding > 100_000_000_000:
        warnings.append("Asset-based: Shares outstanding seems too high")

    return warnings
