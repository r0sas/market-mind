# config.py
"""
Configuration file for Intrinsic Value Calculator
Contains all default parameters and constants used across the application.
"""

# Valuation Model Defaults
DEFAULT_DISCOUNT_RATE = 0.10  # 10%
DEFAULT_TERMINAL_GROWTH = 0.025  # 2.5%
DEFAULT_PROJECTION_YEARS = 5
DEFAULT_BOND_YIELD = 0.0521  # 5.21%
DEFAULT_MARGIN_OF_SAFETY = 0.25  # 25%

# Risk Thresholds
MAX_REASONABLE_GROWTH_RATE = 0.50  # 50% CAGR is suspiciously high
MAX_REASONABLE_PE = 200  # P/E ratios above this are likely outliers
MIN_YEARS_FOR_VALUATION = 2  # Minimum years of data needed

# Data Quality
MIN_HISTORICAL_YEARS = 3  # Warn if less than this
CACHE_TTL_SECONDS = 3600  # 1 hour cache for stock data

# Essential metrics for intrinsic value calculation
ESSENTIAL_METRICS = [
    'Date',
    'Ticker',
    'Market Cap',
    'P/E Ratio',
    'Basic EPS',
    'Free Cash Flow',
    'Operating Cash Flow',
    'Investing Cash Flow',
    'Annual Dividends',
    'Net Income',
    'Diluted EPS',
    'Share Price',
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Total Equity Gross Minority Interest',
    'Basic Average Shares',
    'Diluted Average Shares',
    'Current Assets',
    'Current Liabilities'
]

# Balance sheet items to fetch
BALANCE_SHEET_ITEMS = [
    'Total Assets', 
    'Total Liabilities Net Minority Interest', 
    'Total Equity Gross Minority Interest', 
    'Current Assets', 
    'Current Liabilities'
]

# Cash flow items to fetch
CASH_FLOW_ITEMS = [
    'Operating Cash Flow', 
    'Investing Cash Flow', 
    'Financing Cash Flow', 
    'Free Cash Flow'
]

# EPS column names to search for
EPS_COLUMNS = ["Basic EPS", "Diluted EPS", "Earnings Per Share"]

# Model display names
MODEL_DISPLAY_NAMES = {
    'dcf': 'DCF Intrinsic Value',
    'ddm_single_stage': 'DDM Single-Stage Value',
    'ddm_multi_stage': 'DDM Multi-Stage Value',
    'pe_model': 'P/E Model Intrinsic Value',
    'asset_based': 'Asset-Based Value',
    'graham_value': 'Modern Graham Value'
}

# Confidence scoring thresholds
CONFIDENCE_THRESHOLDS = {
    'fcf_volatility_high': 0.5,  # Coefficient of variation
    'fcf_volatility_medium': 0.3,
    'pe_outlier_threshold': 3.0,  # Standard deviations from mean
}

# AI Insights Configuration
ENABLE_AI_INSIGHTS = True  # Set to False to disable AI features
GROQ_API_KEY = None  # Will be read from environment variable
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, free Groq model

# Model Selection Thresholds
MODEL_SELECTION_THRESHOLDS = {
    'ddm_single': {
        'min_dividend_years': 5,
        'max_dividend_cagr': 0.07,  # 7% (stricter - was 8%)
        'min_payout_ratio': 0.30,
        'max_payout_ratio': 0.70,
        'min_dividend_consistency': 0.7
    },
    'ddm_multi': {
        'min_dividend_years': 3,
        'min_dividend_cagr': 0.08,
        'min_eps_growth': 0.0
    },
    'dcf': {
        'min_fcf_positive_years': 3,
        'min_fcf_positive_ratio': 0.6,
        'min_revenue_cagr': -0.05,
        'max_payout_ratio': 0.80
    },
    'pe_model': {
        'min_eps_positive_years': 3,
        'min_eps_positive_ratio': 0.6,
        'min_pe': 5,
        'max_pe': 50,
        'max_pe_volatility': 1.5
    },
    'graham': {
        'min_eps_years': 5,
        'min_roe': 0.10,
        'max_debt_equity': 0.5,
        'min_eps_positive_years': 5
    },
    'asset_based': {
        'min_asset_quality': 0.3,
        'min_tangible_book': 0.0,
        'use_when_margin_below': 0.05
    }
}

# Fit score weights for different criteria
FIT_SCORE_WEIGHTS = {
    'primary': 0.50,        # Primary criteria (e.g., "has dividends" for DDM)
    'supporting': 0.30,     # Supporting criteria (e.g., "stable payout")
    'data_quality': 0.20    # Data availability and quality
}