from .config import THRESHOLDS, WEIGHTS

class Scorer:
    """
    All _score_* functions and helpers for model scoring.
    """

    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.explanations = {}

    def _score_model(self, model_name: str, conditions: dict) -> float:
        score = 0.0
        explanation = {'primary': [], 'supporting': [], 'data_quality': []}

        for category, metrics in conditions.items():
            for metric, threshold in metrics.items():
                value = self.metrics.get(metric)
                if value is None:
                    explanation[category].append(f"{metric} missing")
                    continue
                if isinstance(threshold, tuple):
                    # If threshold is (min, max)
                    if not threshold[0] <= value <= threshold[1]:
                        explanation[category].append(f"{metric} out of range {threshold}: {value}")
                    else:
                        score += WEIGHTS.get(category, 0.0) / len(metrics)
                else:
                    if value < threshold:
                        explanation[category].append(f"{metric} below {threshold}: {value}")
                    else:
                        score += WEIGHTS.get(category, 0.0) / len(metrics)

        self.explanations[model_name] = explanation
        return min(score, 1.0)

    # ---------------- Score Methods ---------------- #
    def score_ddm_single(self):
        conditions = {
            'primary': {
                'dividend_stability': THRESHOLDS['ddm_single']['min_dividend_consistency'],
                'num_dividend_years': THRESHOLDS['ddm_single']['min_dividend_years']
            },
            'supporting': {
                'dividend_cagr': THRESHOLDS['ddm_single']['max_dividend_cagr'],
                'payout_ratio': THRESHOLDS['ddm_single']['min_payout_ratio']
            },
            'data_quality': {
                'has_dividends': 1
            }
        }
        return self._score_model('ddm_single', conditions)

    def score_ddm_multi(self):
        conditions = {
            'primary': {
                'num_dividend_years': THRESHOLDS['ddm_multi']['min_dividend_years'],
                'dividend_cagr': THRESHOLDS['ddm_multi']['min_dividend_cagr']
            },
            'supporting': {
                'eps_cagr': THRESHOLDS['ddm_multi']['min_eps_growth']
            },
            'data_quality': {
                'has_dividends': 1
            }
        }
        return self._score_model('ddm_multi', conditions)

    def score_dcf(self):
        conditions = {
            'primary': {
                'fcf_positive_years': THRESHOLDS['dcf']['min_fcf_positive_years'],
                'fcf_positive_ratio': THRESHOLDS['dcf']['min_fcf_positive_ratio']
            },
            'supporting': {
                'revenue_cagr': THRESHOLDS['dcf']['min_revenue_cagr'],
                'payout_ratio': THRESHOLDS['dcf']['max_payout_ratio']
            },
            'data_quality': {
                'has_fcf_data': 1
            }
        }
        return self._score_model('dcf', conditions)

    def score_pe_model(self):
        conditions = {
            'primary': {
                'eps_positive_years': THRESHOLDS['pe_model']['min_eps_positive_years'],
                'pe_ratio': THRESHOLDS['pe_model']['min_pe']
            },
            'supporting': {
                'pe_volatility': THRESHOLDS['pe_model']['max_pe_volatility']
            },
            'data_quality': {
                'has_pe_data': 1
            }
        }
        return self._score_model('pe_model', conditions)

    def score_graham(self):
        conditions = {
            'primary': {
                'eps_positive_years': THRESHOLDS['graham']['min_eps_positive_years'],
                'roe': THRESHOLDS['graham']['min_roe']
            },
            'supporting': {
                'de_ratio': THRESHOLDS['graham']['max_debt_equity']
            },
            'data_quality': {
                'has_balance_sheet': 1
            }
        }
        return self._score_model('graham', conditions)

    def score_asset_based(self):
        conditions = {
            'primary': {
                'asset_quality': THRESHOLDS['asset_based']['min_asset_quality'],
                'tangible_book_ratio': THRESHOLDS['asset_based']['min_tangible_book']
            },
            'supporting': {},
            'data_quality': {
                'has_assets_data': 1
            }
        }
        return self._score_model('asset_based', conditions)
