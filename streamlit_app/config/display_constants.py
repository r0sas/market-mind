from core.config import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_MARGIN_OF_SAFETY,
    MODEL_DISPLAY_NAMES
)

# Color schemes
CONFIDENCE_COLORS = {
    'High': '#28a745',
    'Medium': '#ffc107',
    'Low': '#dc3545',
    'N/A': '#6c757d'
}

STATUS_COLORS = {
    'Undervalued': '#d4edda',
    'Overvalued': '#f8d7da',
    'Fair Value': '#e7f3ff'
}

# Chart colors
CHART_COLORS = {
    'primary': '#1f77b4',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8'
}

# Export defaults
DEFAULT_EXPORT_SETTINGS = {
    'include_warnings': True,
    'include_ai_insights': True,
    'include_competitive': True,
    'include_ml_predictions': True
}

