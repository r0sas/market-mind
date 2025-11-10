"""Display pages and sections"""

from .results_display import display_results
from .ai_insights_display import display_ai_insights
from .competitive_display import display_competitive_comparison
from .ml_predictions_display import display_ml_predictions
from .margin_of_safety_display import display_margin_of_safety
from .sensitivity_display import display_sensitivity_analysis
from .model_selection_display import display_model_selection_analysis
from .information_sections import render_information_sections

__all__ = [
    'display_results',
    'display_ai_insights',
    'display_competitive_comparison',
    'display_ml_predictions',
    'display_margin_of_safety',
    'display_sensitivity_analysis',
    'display_model_selection_analysis',
    'render_information_sections'
]

