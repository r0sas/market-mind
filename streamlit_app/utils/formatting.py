from typing import Dict
from config.display_constants import CONFIDENCE_COLORS

def format_currency(value: float) -> str:
    """Format value as currency"""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"

def format_large_number(value: float) -> str:
    """Format large numbers with B/M suffixes"""
    if value is None:
        return "N/A"
    
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def create_confidence_badge(confidence: str) -> str:
    """Create HTML badge for confidence level"""
    color = CONFIDENCE_COLORS.get(confidence, CONFIDENCE_COLORS['N/A'])
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:3px;font-size:0.8em">{confidence}</span>'

def get_combined_signal(iv_diff_pct: float, ml_direction: str) -> Dict[str, str]:
    """
    Combine intrinsic value and ML prediction into actionable signal.
    
    Args:
        iv_diff_pct: Percentage difference between IV and current price
        ml_direction: ML prediction direction ('UP ⬆️', 'DOWN ⬇️', or 'FLAT ➡️')
        
    Returns:
        Dict with signal, color, and explanation
    """
    iv_bullish = iv_diff_pct > 10
    iv_bearish = iv_diff_pct < -10
    ml_bullish = 'UP' in str(ml_direction)
    ml_bearish = 'DOWN' in str(ml_direction)
    
    if iv_bullish and ml_bullish:
        return {
            'signal': '🚀 STRONG BUY',
            'color': 'success',
            'explanation': 'Both valuation and ML models indicate strong upside potential'
        }
    elif iv_bearish and ml_bearish:
        return {
            'signal': '🛑 STRONG SELL',
            'color': 'error',
            'explanation': 'Both valuation and ML models indicate strong downside risk'
        }
    elif iv_bullish and ml_bearish:
        return {
            'signal': '⚠️ CAUTIOUS',
            'color': 'warning',
            'explanation': 'Undervalued but ML predicts decline - value trap risk'
        }
    elif iv_bearish and ml_bullish:
        return {
            'signal': '⚠️ MOMENTUM',
            'color': 'warning',
            'explanation': 'Overvalued but ML predicts rise - momentum trade opportunity'
        }
    elif iv_bullish:
        return {
            'signal': '✅ BUY',
            'color': 'success',
            'explanation': 'Undervalued with neutral ML outlook'
        }
    elif iv_bearish:
        return {
            'signal': '❌ SELL',
            'color': 'error',
            'explanation': 'Overvalued with neutral ML outlook'
        }
    else:
        return {
            'signal': '➡️ HOLD',
            'color': 'info',
            'explanation': 'No strong signal - stock fairly valued'
        }

