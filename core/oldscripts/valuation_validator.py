# valuation_validator.py
"""
Valuation Result Validator
Detects and flags unrealistic valuation results
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValuationValidator:
    """
    Validates valuation results and flags suspicious values.
    
    Detects:
    - Unrealistic prices (too high/low)
    - Unit conversion errors (Yen, Won, etc.)
    - Shares outstanding errors
    - Extreme deviations from current price
    """
    
    # Reasonable bounds for stock prices (in USD)
    MIN_REASONABLE_PRICE = 0.01  # $0.01 (penny stocks)
    MAX_REASONABLE_PRICE = 10000  # $10,000 (even BRK.A is ~$600k, but single stocks rarely exceed $10k)
    
    # Extreme deviation thresholds
    MAX_DEVIATION_FROM_CURRENT = 50  # 50x current price is suspicious
    MIN_DEVIATION_FROM_CURRENT = 0.01  # 1/100th of current price is suspicious
    
    # Known foreign currency stocks that need special handling
    FOREIGN_TICKERS = {
        'TM': 'JPY',      # Toyota - Japanese Yen
        '7203.T': 'JPY',  # Toyota Japan listing
        '005930.KS': 'KRW',  # Samsung - Korean Won
        'BABA': 'USD',    # Alibaba - USD (ADR)
        'TSM': 'USD',     # TSMC - USD (ADR)
    }
    
    def __init__(self, ticker: str, current_price: float):
        """
        Initialize validator.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current market price
        """
        self.ticker = ticker
        self.current_price = current_price
        self.warnings: List[str] = []
        self.corrections: Dict[str, float] = {}
    
    def validate_valuation(
        self, 
        model_name: str, 
        calculated_value: float,
        total_assets: Optional[float] = None,
        total_liabilities: Optional[float] = None,
        shares_outstanding: Optional[float] = None
    ) -> Tuple[bool, Optional[float], List[str]]:
        """
        Validate a single valuation result.
        
        Args:
            model_name: Name of the valuation model
            calculated_value: The calculated intrinsic value
            total_assets: Total assets (for asset-based validation)
            total_liabilities: Total liabilities (for asset-based validation)
            shares_outstanding: Shares outstanding (for validation)
            
        Returns:
            Tuple of (is_valid, corrected_value, warnings)
        """
        warnings = []
        corrected_value = calculated_value
        is_valid = True
        
        # Check 1: Absolute price bounds
        if calculated_value < self.MIN_REASONABLE_PRICE:
            warnings.append(
                f"{model_name}: Value ${calculated_value:.4f} is unreasonably low (< ${self.MIN_REASONABLE_PRICE})"
            )
            is_valid = False
        
        if calculated_value > self.MAX_REASONABLE_PRICE:
            warnings.append(
                f"{model_name}: Value ${calculated_value:,.2f} is unreasonably high (> ${self.MAX_REASONABLE_PRICE:,})"
            )
            is_valid = False
            
            # Check for unit conversion error
            corrected_value, corrected = self._check_currency_conversion_error(
                calculated_value, model_name
            )
            if corrected:
                warnings.append(
                    f"{model_name}: Possible currency conversion error detected. "
                    f"Corrected from ${calculated_value:,.2f} to ${corrected_value:.2f}"
                )
                is_valid = True  # It's valid after correction
        
        # Check 2: Deviation from current price
        if self.current_price and self.current_price > 0:
            ratio = calculated_value / self.current_price
            
            if ratio > self.MAX_DEVIATION_FROM_CURRENT:
                warnings.append(
                    f"{model_name}: Value is {ratio:.1f}x current price - extremely high"
                )
                # Check if it's a shares outstanding error
                if shares_outstanding:
                    corrected_value, corrected = self._check_shares_error(
                        calculated_value, total_assets, total_liabilities, shares_outstanding, model_name
                    )
                    if corrected:
                        warnings.append(
                            f"{model_name}: Possible shares outstanding error. "
                            f"Corrected from ${calculated_value:,.2f} to ${corrected_value:.2f}"
                        )
                        is_valid = True
                    else:
                        is_valid = False
                else:
                    is_valid = False
            
            elif ratio < self.MIN_DEVIATION_FROM_CURRENT:
                warnings.append(
                    f"{model_name}: Value is {1/ratio:.1f}x smaller than current price - extremely low"
                )
                is_valid = False
        
        # Check 3: Model-specific validations
        if model_name.lower() in ['asset_based', 'asset-based', 'book value']:
            asset_warnings = self._validate_asset_based(
                calculated_value, total_assets, total_liabilities, shares_outstanding
            )
            warnings.extend(asset_warnings)
            
            # If asset-based gives extreme value, it's likely an error
            if len(asset_warnings) > 0 and calculated_value > 1000:
                is_valid = False
        
        return is_valid, corrected_value if is_valid else None, warnings
    
    def _check_currency_conversion_error(
        self, 
        value: float, 
        model_name: str
    ) -> Tuple[float, bool]:
        """
        Check if value might be in foreign currency.
        
        Common errors:
        - Japanese Yen (JPY): ~100-150 per USD
        - Korean Won (KRW): ~1,200-1,400 per USD
        """
        # Check if this is a known foreign ticker
        if self.ticker in self.FOREIGN_TICKERS:
            currency = self.FOREIGN_TICKERS[self.ticker]
            
            if currency == 'JPY' and value > 1000:
                # Likely in Yen, convert to USD
                corrected = value / 150  # Approximate JPY/USD rate
                return corrected, True
            
            elif currency == 'KRW' and value > 10000:
                # Likely in Won, convert to USD
                corrected = value / 1300  # Approximate KRW/USD rate
                return corrected, True
        
        # Check if value looks like it might be in Yen
        if value > 10000 and value < 100000:
            # Could be Yen
            corrected = value / 150
            if abs(corrected - self.current_price) / self.current_price < 0.5:
                # Correction brings it close to current price
                logger.warning(f"{model_name} for {self.ticker}: Detected possible JPY value, converting to USD")
                return corrected, True
        
        return value, False
    
    def _check_shares_error(
        self,
        value: float,
        total_assets: Optional[float],
        total_liabilities: Optional[float],
        shares_outstanding: float,
        model_name: str
    ) -> Tuple[float, bool]:
        """
        Check if shares outstanding might be in wrong units.
        
        Common error: Shares in billions but treated as actual shares
        """
        if not total_assets or not total_liabilities:
            return value, False
        
        # Calculate what shares would need to be to get reasonable value
        book_value = total_assets - total_liabilities
        
        # If current shares give extreme value, try different units
        if value > 1000:
            # Try shares * 1000 (if reported in thousands)
            corrected_shares = shares_outstanding * 1000
            corrected_value = book_value / corrected_shares
            
            if self.MIN_REASONABLE_PRICE < corrected_value < self.MAX_REASONABLE_PRICE:
                logger.warning(
                    f"{model_name} for {self.ticker}: Shares might be in thousands. "
                    f"Correcting from {shares_outstanding:,.0f} to {corrected_shares:,.0f}"
                )
                return corrected_value, True
            
            # Try shares * 1,000,000 (if reported in millions)
            corrected_shares = shares_outstanding * 1_000_000
            corrected_value = book_value / corrected_shares
            
            if self.MIN_REASONABLE_PRICE < corrected_value < self.MAX_REASONABLE_PRICE:
                logger.warning(
                    f"{model_name} for {self.ticker}: Shares might be in millions. "
                    f"Correcting from {shares_outstanding:,.0f} to {corrected_shares:,.0f}"
                )
                return corrected_value, True
        
        return value, False
    
    def _validate_asset_based(
        self,
        calculated_value: float,
        total_assets: Optional[float],
        total_liabilities: Optional[float],
        shares_outstanding: Optional[float]
    ) -> List[str]:
        """Validate asset-based model specifically."""
        warnings = []
        
        if not all([total_assets, total_liabilities, shares_outstanding]):
            warnings.append("Asset-based: Missing balance sheet data for validation")
            return warnings
        
        # Calculate expected book value
        book_value = total_assets - total_liabilities
        expected_per_share = book_value / shares_outstanding
        
        # Check if calculated matches expected
        if abs(calculated_value - expected_per_share) / expected_per_share > 0.01:
            warnings.append(
                f"Asset-based: Calculated ${calculated_value:,.2f} doesn't match "
                f"expected ${expected_per_share:,.2f} (Book Value / Shares)"
            )
        
        # Check for negative book value
        if book_value < 0:
            warnings.append("Asset-based: Company has negative book value (liabilities > assets)")
        
        # Check shares outstanding magnitude
        if shares_outstanding < 1000:
            warnings.append(
                f"Asset-based: Shares outstanding ({shares_outstanding:,.0f}) seems too low - "
                "might be in wrong units (thousands/millions)"
            )
        
        if shares_outstanding > 100_000_000_000:  # 100 billion
            warnings.append(
                f"Asset-based: Shares outstanding ({shares_outstanding:,.0f}) seems too high"
            )
        
        return warnings
    
    def validate_all(
        self, 
        valuations: Dict[str, float],
        financial_data: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Validate all valuation results.
        
        Args:
            valuations: Dict of {model_name: value}
            financial_data: Optional dict with balance sheet data
            
        Returns:
            Dict with validated results and warnings
        """
        validated = {}
        all_warnings = []
        corrected_values = {}
        
        for model_name, value in valuations.items():
            is_valid, corrected, warnings = self.validate_valuation(
                model_name=model_name,
                calculated_value=value,
                total_assets=financial_data.get('total_assets') if financial_data else None,
                total_liabilities=financial_data.get('total_liabilities') if financial_data else None,
                shares_outstanding=financial_data.get('shares_outstanding') if financial_data else None
            )
            
            validated[model_name] = {
                'original_value': value,
                'is_valid': is_valid,
                'corrected_value': corrected,
                'warnings': warnings
            }
            
            all_warnings.extend(warnings)
            
            if corrected and corrected != value:
                corrected_values[model_name] = corrected
        
        return {
            'validated': validated,
            'all_warnings': all_warnings,
            'corrections': corrected_values,
            'has_errors': any(not v['is_valid'] for v in validated.values())
        }


def validate_and_correct_valuations(
    ticker: str,
    current_price: float,
    valuations: Dict[str, float],
    financial_data: Optional[Dict] = None
) -> Tuple[Dict[str, float], List[str]]:
    """
    Convenience function to validate and auto-correct valuations.
    
    Args:
        ticker: Stock ticker
        current_price: Current market price
        valuations: Dict of {model: value}
        financial_data: Optional balance sheet data
        
    Returns:
        Tuple of (corrected_valuations, warnings)
    """
    validator = ValuationValidator(ticker, current_price)
    result = validator.validate_all(valuations, financial_data)
    
    # Apply corrections
    corrected_valuations = valuations.copy()
    for model, corrected_value in result['corrections'].items():
        corrected_valuations[model] = corrected_value
    
    # Remove invalid valuations
    for model, validation in result['validated'].items():
        if not validation['is_valid'] and validation['corrected_value'] is None:
            logger.warning(f"Removing invalid valuation for {ticker} {model}: {validation['original_value']}")
            corrected_valuations.pop(model, None)
    
    return corrected_valuations, result['all_warnings']


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VALUATION VALIDATOR TEST")
    print("="*70)
    
    # Test Case 1: Toyota (TM) - suspected Yen conversion error
    print("\n[Test 1] Toyota (TM) - Asset-Based Model")
    print("-" * 70)
    
    tm_valuations = {
        'dcf': 185.50,
        'pe_model': 195.20,
        'asset_based': 27000.00  # SUSPICIOUS!
    }
    
    tm_financial = {
        'total_assets': 68_500_000_000_000,  # 68.5 trillion (might be in Yen?)
        'total_liabilities': 59_000_000_000_000,
        'shares_outstanding': 13_500_000_000  # 13.5 billion shares
    }
    
    corrected, warnings = validate_and_correct_valuations(
        ticker='TM',
        current_price=180.00,
        valuations=tm_valuations,
        financial_data=tm_financial
    )
    
    print("Original valuations:")
    for model, value in tm_valuations.items():
        print(f"  {model}: ${value:,.2f}")
    
    print("\nCorrected valuations:")
    for model, value in corrected.items():
        print(f"  {model}: ${value:,.2f}")
    
    print("\nWarnings:")
    for warning in warnings:
        print(f"  ⚠️  {warning}")
    
    # Test Case 2: Normal stock (AAPL)
    print("\n" + "="*70)
    print("[Test 2] Apple (AAPL) - Normal Values")
    print("-" * 70)
    
    aapl_valuations = {
        'dcf': 195.00,
        'ddm': 185.00,
        'pe_model': 200.00,
        'asset_based': 75.50
    }
    
    corrected_aapl, warnings_aapl = validate_and_correct_valuations(
        ticker='AAPL',
        current_price=180.00,
        valuations=aapl_valuations
    )
    
    print("Valuations:")
    for model, value in corrected_aapl.items():
        status = "✅" if value == aapl_valuations[model] else "🔧"
        print(f"  {status} {model}: ${value:,.2f}")
    
    if warnings_aapl:
        print("\nWarnings:")
        for warning in warnings_aapl:
            print(f"  ⚠️  {warning}")
    else:
        print("\n✅ All valuations passed validation!")
    
    print("\n" + "="*70)