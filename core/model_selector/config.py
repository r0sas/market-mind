# ---------------------- Thresholds ---------------------- #
THRESHOLDS = {
    'ddm_single': {
        'min_dividend_consistency': 0.8,
        'min_dividend_years': 5,
        'max_dividend_cagr': 0.15,
        'min_payout_ratio': 0.3,
    },
    'ddm_multi': {
        'min_dividend_years': 5,
        'min_dividend_cagr': 0.05,
        'min_eps_growth': 0.05,
    },
    'dcf': {
        'min_fcf_positive_years': 3,
        'min_fcf_positive_ratio': 0.7,
        'min_revenue_cagr': 0.05,
        'max_payout_ratio': 0.7,
    },
    'pe_model': {
        'min_eps_positive_years': 3,
        'min_pe': 5,
        'max_pe_volatility': 0.3,
    },
    'graham': {
        'min_eps_positive_years': 3,
        'min_roe': 0.12,
        'max_debt_equity': 1.0,
    },
    'asset_based': {
        'min_asset_quality': 0.7,
        'min_tangible_book': 0.5,
    },
}

# ---------------------- Weights ---------------------- #
WEIGHTS = {
    'primary': 0.5,
    'supporting': 0.3,
    'data_quality': 0.2,
}

# ---------------------- Model Names ---------------------- #
MODEL_DISPLAY_NAMES = {
    'ddm_single': 'DDM Single',
    'ddm_multi': 'DDM Multi',
    'dcf': 'Discounted Cash Flow',
    'pe_model': 'P/E Model',
    'graham': 'Graham Model',
    'asset_based': 'Asset-Based Model',
}
