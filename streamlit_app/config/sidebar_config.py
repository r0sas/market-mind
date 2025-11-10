"""Sidebar configuration constants"""

# Slider ranges
DISCOUNT_RATE_RANGE = (5.0, 20.0)
TERMINAL_GROWTH_RANGE = (0.0, 5.0)
MARGIN_OF_SAFETY_RANGE = (10, 50)
FIT_SCORE_RANGE = (0.3, 0.9)

# Defaults
DEFAULT_FIT_SCORE = 0.5
DEFAULT_SHOW_ALL_SCORES = False
DEFAULT_SHOW_CONFIDENCE = True
DEFAULT_SHOW_WARNINGS = True
DEFAULT_USE_WEIGHTED_AVG = False

# Help texts
HELP_TEXTS = {
    'auto_select': 'AI analyzes company data and selects best models automatically',
    'fit_score': 'Higher scores = better model fit to company data',
    'weighted_avg': 'Weights models by confidence scores',
    'margin_safety': 'Safety buffer below intrinsic value for buy price',
    'sensitivity': 'Shows how valuation changes with different parameters'
}

