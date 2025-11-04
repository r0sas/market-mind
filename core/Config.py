# core/Config.py
"""
Configuration settings for the Intrinsic Value Calculator
"""

# ============================================
# MODEL DISPLAY NAMES
# ============================================

MODEL_DISPLAY_NAMES = {
    'dcf': 'DCF (Discounted Cash Flow)',
    'ddm_single_stage': 'DDM Single-Stage',
    'ddm_multi_stage': 'DDM Multi-Stage',
    'pe_model': 'P/E Multiplier Model',
    'graham_value': 'Modern Graham Formula',
    'asset_based': 'Asset-Based Valuation'
}

# ============================================
# DEFAULT PARAMETERS
# ============================================

DEFAULT_DISCOUNT_RATE = 0.10  # 10%
DEFAULT_TERMINAL_GROWTH = 0.025  # 2.5%
DEFAULT_MARGIN_OF_SAFETY = 0.25  # 25%
MIN_HISTORICAL_YEARS = 5
ENABLE_AI_INSIGHTS = False  # Set to True if you have Groq API key

# ============================================
# DATA FETCHER CONFIGURATION
# ============================================

# Income Statement Columns
EPS_COLUMNS = ['Diluted EPS', 'Basic EPS']

# Balance Sheet Items
BALANCE_SHEET_ITEMS = [
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Total Equity Gross Minority Interest',
    'Current Assets',
    'Current Liabilities',
    'Total Debt',
    'Long Term Debt',
    'Cash And Cash Equivalents',
    'Stockholders Equity',
]

# Cash Flow Items
CASH_FLOW_ITEMS = [
    'Free Cash Flow',
    'Operating Cash Flow',
    'Investing Cash Flow',
    'Financing Cash Flow',
    'Capital Expenditure',
]

# ============================================
# ESSENTIAL METRICS (for IV Calculator)
# ============================================

ESSENTIAL_METRICS = [
    # Core Valuation Metrics
    'Free Cash Flow',
    'Operating Cash Flow',
    'Investing Cash Flow',
    'Annual Dividends',
    'P/E Ratio',
    'Diluted EPS',
    'Basic EPS',
    'Share Price',
    
    # Balance Sheet
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Total Equity Gross Minority Interest',
    
    # Shares
    'Basic Average Shares',
    'Diluted Average Shares',
    
    # Income
    'Net Income',
    
    # Identifiers
    'Ticker',
]

# ============================================
# ML-ENHANCED METRICS (for Machine Learning)
# ============================================

ESSENTIAL_METRICS_ML = [
    # === All original essential metrics ===
    'Free Cash Flow',
    'Operating Cash Flow',
    'Investing Cash Flow',
    'Annual Dividends',
    'P/E Ratio',
    'Diluted EPS',
    'Basic EPS',
    'Share Price',
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Total Equity Gross Minority Interest',
    'Basic Average Shares',
    'Diluted Average Shares',
    'Net Income',
    'Ticker',
    
    # === Additional Profitability Metrics ===
    'Operating Income',
    'Gross Profit',
    'EBITDA',
    'Total Revenue',
    
    # === Additional Valuation Metrics ===
    'Enterprise Value',
    'Market Cap',
    'Current Price',
    
    # === Additional Balance Sheet Metrics ===
    'Current Assets',
    'Current Liabilities',
    'Total Debt',
    'Long Term Debt',
    'Cash And Cash Equivalents',
    'Stockholders Equity',
    
    # === Additional Cash Flow Metrics ===
    'Capital Expenditure',
    'Financing Cash Flow',
    
    # === Company Information ===
    'Sector',
    'Industry',
]

# ============================================
# MODEL SELECTOR CONFIGURATION
# ============================================

# Thresholds for model selection
MODEL_SELECTOR_THRESHOLDS = {
    'min_fcf_positive_years': 3,
    'min_dividend_years': 5,
    'min_eps_positive_years': 3,
    'min_profitable_years': 5,
}