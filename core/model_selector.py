import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from core.Config import MODEL_DISPLAY_NAMES, DEFAULT_DISCOUNT_RATE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Intelligent model selection for intrinsic value calculations.
    
    Analyzes company financials and determines which valuation models
    are most appropriate based on business characteristics, financial
    health, and data availability.
    
    Example:
        >>> selector = ModelSelector(simplified_df)
        >>> fit_scores = selector.calculate_fit_scores()
        >>> recommended = selector.get_recommended_models(min_score=0.5)
        >>> explanations = selector.get_fit_explanations()
    """
    
    # Model selection thresholds
    THRESHOLDS = {
        'ddm_single': {
            'min_dividend_years': 5,
            'max_dividend_cagr': 0.08,  # 8%
            'min_payout_ratio': 0.30,
            'max_payout_ratio': 0.70,
            'min_dividend_consistency': 0.7  # 70% of years must have dividends
        },
        'ddm_multi': {
            'min_dividend_years': 3,
            'min_dividend_cagr': 0.08,  # 8%
            'min_eps_growth': 0.0  # Must be growing
        },
        'dcf': {
            'min_fcf_positive_years': 3,
            'min_fcf_positive_ratio': 0.6,  # 60% of years
            'min_revenue_cagr': -0.05,  # Allow slight decline
            'max_payout_ratio': 0.80  # High payout = less reinvestment
        },
        'pe_model': {
            'min_eps_positive_years': 3,
            'min_eps_positive_ratio': 0.6,
            'min_pe': 5,
            'max_pe': 50,
            'max_pe_volatility': 1.5  # Coefficient of variation
        },
        'graham': {
            'min_eps_years': 5,
            'min_roe': 0.10,
            'max_debt_equity': 0.5,
            'min_eps_positive_years': 5
        },
        'asset_based': {
            'min_asset_quality': 0.3,  # (Assets - Liabilities) / Assets
            'min_tangible_book': 0.0,  # Must have positive book value
            'use_when_margin_below': 0.05  # 5% net margin
        }
    }
    
    # Weights for fit score calculation
    WEIGHTS = {
        'primary': 0.50,      # Primary criteria (e.g., has dividends)
        'supporting': 0.30,   # Supporting criteria (e.g., stable)
        'data_quality': 0.20  # Data availability and quality
    }
    
    def __init__(self, simplified_df: pd.DataFrame):
        """
        Initialize ModelSelector with simplified financial data.
        
        Args:
            simplified_df: DataFrame from IVSimplifier
        """
        if simplified_df is None or simplified_df.empty:
            raise ValueError("simplified_df cannot be None or empty")
        
        self.df = simplified_df.copy()
        self.metrics: Dict[str, any] = {}
        self.fit_scores: Dict[str, float] = {}
        self.explanations: Dict[str, Dict[str, List[str]]] = {}
        self.exclusion_reasons: Dict[str, str] = {}
        
        # Extract metrics needed for selection
        self._extract_metrics()
    
    def _extract_metrics(self) -> None:
        """Extract all metrics needed for model selection."""
        
        # Basic info
        self.metrics['num_years'] = len(self.df.columns)
        
        # Dividend analysis
        self.metrics['has_dividends'] = self._check_dividends()
        self.metrics['dividend_stability'] = self._calculate_dividend_stability()
        self.metrics['dividend_cagr'] = self._calculate_dividend_cagr()
        self.metrics['payout_ratio'] = self._calculate_payout_ratio()
        
        # Cash flow analysis
        self.metrics['fcf_positive_years'] = self._count_positive_years('Free Cash Flow')
        self.metrics['fcf_consistency'] = self._calculate_consistency('Free Cash Flow')
        self.metrics['fcf_cagr'] = self._calculate_cagr('Free Cash Flow')
        
        # Earnings analysis
        self.metrics['eps_positive_years'] = self._count_positive_eps_years()
        self.metrics['eps_stability'] = self._calculate_eps_stability()
        self.metrics['eps_cagr'] = self._calculate_eps_cagr()
        
        # Balance sheet analysis
        self.metrics['asset_quality'] = self._calculate_asset_quality()
        self.metrics['debt_equity'] = self._calculate_debt_equity()
        self.metrics['tangible_book_value'] = self._calculate_tangible_book_value()
        
        # Profitability
        self.metrics['roe'] = self._calculate_roe()
        self.metrics['net_margin'] = self._calculate_net_margin()
        
        # P/E analysis
        self.metrics['pe_volatility'] = self._calculate_pe_volatility()
        self.metrics['avg_pe'] = self._calculate_avg_pe()
        
        logger.info(f"Extracted metrics for model selection: {len(self.metrics)} metrics")
    
    def _check_dividends(self) -> bool:
        """Check if company pays dividends."""
        if 'Annual Dividends' not in self.df.index:
            return False
        divs = self.df.loc['Annual Dividends'].dropna()
        return len(divs) > 0 and divs.sum() > 0
    
    def _calculate_dividend_stability(self) -> float:
        """Calculate dividend stability (1 - coefficient of variation)."""
        if not self.metrics.get('has_dividends', False):
            return 0.0
        
        if 'Annual Dividends' not in self.df.index:
            return 0.0
        
        divs = self.df.loc['Annual Dividends'].dropna()
        divs = divs[divs > 0]  # Only positive dividends
        
        if len(divs) < 2:
            return 0.0
        
        mean_div = divs.mean()
        std_div = divs.std()
        
        if mean_div == 0:
            return 0.0
        
        cv = std_div / mean_div
        stability = max(0, 1 - cv)  # Lower CV = higher stability
        
        return stability
    
    def _calculate_dividend_cagr(self) -> Optional[float]:
        """Calculate dividend CAGR."""
        if not self.metrics.get('has_dividends', False):
            return None
        
        if 'Annual Dividends' not in self.df.index:
            return None
        
        divs = self.df.loc['Annual Dividends'].dropna()
        divs = divs[divs > 0]
        
        if len(divs) < 2:
            return None
        
        div_latest = divs.iloc[0]
        div_earliest = divs.iloc[-1]
        n_years = len(divs) - 1
        
        if div_earliest <= 0:
            return None
        
        cagr = (div_latest / div_earliest) ** (1 / n_years) - 1
        return cagr
    
    def _calculate_payout_ratio(self) -> Optional[float]:
        """Calculate dividend payout ratio (Dividend / EPS)."""
        if not self.metrics.get('has_dividends', False):
            return None
        
        if 'Annual Dividends' not in self.df.index or 'Diluted EPS' not in self.df.index:
            return None
        
        div = self.df.loc['Annual Dividends'].iloc[0]
        eps = self.df.loc['Diluted EPS'].iloc[0] if 'Diluted EPS' in self.df.index else self.df.loc['Basic EPS'].iloc[0]
        
        if eps <= 0:
            return None
        
        payout = div / eps
        return payout
    
    def _count_positive_years(self, metric: str) -> int:
        """Count how many years a metric was positive."""
        if metric not in self.df.index:
            return 0
        
        values = self.df.loc[metric].dropna()
        return (values > 0).sum()
    
    def _calculate_consistency(self, metric: str) -> float:
        """Calculate what percentage of years a metric was positive."""
        if metric not in self.df.index:
            return 0.0
        
        values = self.df.loc[metric].dropna()
        if len(values) == 0:
            return 0.0
        
        return (values > 0).sum() / len(values)
    
    def _calculate_cagr(self, metric: str) -> Optional[float]:
        """Calculate CAGR for a metric."""
        if metric not in self.df.index:
            return None
        
        values = self.df.loc[metric].dropna()
        
        if len(values) < 2:
            return None
        
        latest = values.iloc[0]
        earliest = values.iloc[-1]
        n_years = len(values) - 1
        
        if earliest <= 0 or latest <= 0:
            return None
        
        cagr = (latest / earliest) ** (1 / n_years) - 1
        return cagr
    
    def _count_positive_eps_years(self) -> int:
        """Count years with positive EPS."""
        eps_metric = 'Diluted EPS' if 'Diluted EPS' in self.df.index else 'Basic EPS'
        
        if eps_metric not in self.df.index:
            return 0
        
        eps = self.df.loc[eps_metric].dropna()
        return (eps > 0).sum()
    
    def _calculate_eps_stability(self) -> float:
        """Calculate EPS stability (1 - coefficient of variation)."""
        eps_metric = 'Diluted EPS' if 'Diluted EPS' in self.df.index else 'Basic EPS'
        
        if eps_metric not in self.df.index:
            return 0.0
        
        eps = self.df.loc[eps_metric].dropna()
        eps = eps[eps > 0]  # Only positive EPS
        
        if len(eps) < 2:
            return 0.0
        
        mean_eps = eps.mean()
        std_eps = eps.std()
        
        if mean_eps == 0:
            return 0.0
        
        cv = std_eps / mean_eps
        stability = max(0, 1 - cv)
        
        return stability
    
    def _calculate_eps_cagr(self) -> Optional[float]:
        """Calculate EPS CAGR."""
        eps_metric = 'Diluted EPS' if 'Diluted EPS' in self.df.index else 'Basic EPS'
        return self._calculate_cagr(eps_metric)
    
    def _calculate_asset_quality(self) -> Optional[float]:
        """Calculate asset quality: (Assets - Liabilities) / Assets."""
        if 'Total Assets' not in self.df.index or 'Total Liabilities Net Minority Interest' not in self.df.index:
            return None
        
        assets = self.df.loc['Total Assets'].iloc[0]
        liabilities = self.df.loc['Total Liabilities Net Minority Interest'].iloc[0]
        
        if assets <= 0:
            return None
        
        quality = (assets - liabilities) / assets
        return quality
    
    def _calculate_debt_equity(self) -> Optional[float]:
        """Calculate debt-to-equity ratio."""
        if 'Total Liabilities Net Minority Interest' not in self.df.index or 'Total Equity Gross Minority Interest' not in self.df.index:
            return None
        
        liabilities = self.df.loc['Total Liabilities Net Minority Interest'].iloc[0]
        equity = self.df.loc['Total Equity Gross Minority Interest'].iloc[0]
        
        if equity <= 0:
            return None
        
        debt_equity = liabilities / equity
        return debt_equity
    
    def _calculate_tangible_book_value(self) -> Optional[float]:
        """Calculate tangible book value per share."""
        if 'Total Assets' not in self.df.index or 'Total Liabilities Net Minority Interest' not in self.df.index:
            return None
        
        assets = self.df.loc['Total Assets'].iloc[0]
        liabilities = self.df.loc['Total Liabilities Net Minority Interest'].iloc[0]
        
        # Get shares outstanding
        shares = None
        if 'Basic Average Shares' in self.df.index:
            shares = self.df.loc['Basic Average Shares'].iloc[0]
        elif 'Diluted Average Shares' in self.df.index:
            shares = self.df.loc['Diluted Average Shares'].iloc[0]
        
        if shares is None or shares <= 0:
            return None
        
        book_value = (assets - liabilities) / shares
        return book_value
    
    def _calculate_roe(self) -> Optional[float]:
        """Calculate Return on Equity."""
        if 'Net Income' not in self.df.index or 'Total Equity Gross Minority Interest' not in self.df.index:
            return None
        
        net_income = self.df.loc['Net Income'].iloc[0]
        equity = self.df.loc['Total Equity Gross Minority Interest'].iloc[0]
        
        if equity <= 0:
            return None
        
        roe = net_income / equity
        return roe
    
    def _calculate_net_margin(self) -> Optional[float]:
        """Calculate net profit margin."""
        if 'Net Income' not in self.df.index or 'Total Revenue' not in self.df.index:
            return None
        
        net_income = self.df.loc['Net Income'].iloc[0]
        revenue = self.df.loc['Total Revenue'].iloc[0]
        
        if revenue <= 0:
            return None
        
        margin = net_income / revenue
        return margin
    
    def _calculate_pe_volatility(self) -> Optional[float]:
        """Calculate P/E ratio coefficient of variation."""
        if 'P/E Ratio' not in self.df.index:
            return None
        
        pe = self.df.loc['P/E Ratio'].dropna()
        pe = pe[(pe > 0) & (pe < 200)]  # Filter outliers
        
        if len(pe) < 2:
            return None
        
        mean_pe = pe.mean()
        std_pe = pe.std()
        
        if mean_pe == 0:
            return None
        
        cv = std_pe / mean_pe
        return cv
    
    def _calculate_avg_pe(self) -> Optional[float]:
        """Calculate average P/E ratio."""
        if 'P/E Ratio' not in self.df.index:
            return None
        
        pe = self.df.loc['P/E Ratio'].dropna()
        pe = pe[(pe > 0) & (pe < 200)]  # Filter outliers
        
        if len(pe) == 0:
            return None
        
        return pe.median()  # Use median to reduce impact of outliers
    
    def calculate_fit_scores(self) -> Dict[str, float]:
        """
        Calculate fit scores for all valuation models.
        
        Returns:
            Dictionary with model names and fit scores (0-1)
        """
        self.fit_scores = {
            'dcf': self._score_dcf(),
            'ddm_single_stage': self._score_ddm_single(),
            'ddm_multi_stage': self._score_ddm_multi(),
            'pe_model': self._score_pe_model(),
            'graham_value': self._score_graham(),
            'asset_based': self._score_asset_based()
        }
        
        logger.info(f"Calculated fit scores: {self.fit_scores}")
        return self.fit_scores.copy()
    
    def _score_dcf(self) -> float:
        """Calculate fit score for DCF model."""
        reasons_pass = []
        reasons_fail = []
        
        primary_score = 0.0
        supporting_score = 0.0
        data_quality_score = 0.0
        
        # Primary: Positive FCF
        fcf_positive_years = self.metrics.get('fcf_positive_years', 0)
        fcf_consistency = self.metrics.get('fcf_consistency', 0)
        
        if fcf_positive_years >= self.THRESHOLDS['dcf']['min_fcf_positive_years']:
            primary_score = min(1.0, fcf_consistency / self.THRESHOLDS['dcf']['min_fcf_positive_ratio'])
            reasons_pass.append(f"Positive FCF in {fcf_positive_years}/{self.metrics['num_years']} years")
        else:
            reasons_fail.append(f"Insufficient positive FCF years ({fcf_positive_years}/{self.THRESHOLDS['dcf']['min_fcf_positive_years']} required)")
        
        # Supporting: FCF growth
        fcf_cagr = self.metrics.get('fcf_cagr')
        if fcf_cagr is not None and fcf_cagr > 0:
            supporting_score += 0.5
            reasons_pass.append(f"Positive FCF growth ({fcf_cagr*100:.1f}% CAGR)")
        elif fcf_cagr is not None:
            supporting_score += 0.2
            reasons_pass.append(f"FCF CAGR: {fcf_cagr*100:.1f}%")
        
        # Supporting: Moderate payout (retaining cash for growth)
        payout = self.metrics.get('payout_ratio')
        if payout is not None and payout < self.THRESHOLDS['dcf']['max_payout_ratio']:
            supporting_score += 0.5
            reasons_pass.append(f"Moderate payout ratio ({payout*100:.1f}%)")
        elif payout is not None:
            supporting_score += 0.2
            reasons_fail.append(f"High payout ratio ({payout*100:.1f}%) - less reinvestment")
        else:
            supporting_score += 0.3  # No dividend is okay for DCF
        
        # Normalize supporting score
        supporting_score = min(1.0, supporting_score)
        
        # Data quality: Years of history
        if self.metrics['num_years'] >= 5:
            data_quality_score = 1.0
        elif self.metrics['num_years'] >= 3:
            data_quality_score = 0.7
        else:
            data_quality_score = 0.4
            reasons_fail.append(f"Limited history ({self.metrics['num_years']} years)")
        
        # Calculate weighted score
        total_score = (
            primary_score * self.WEIGHTS['primary'] +
            supporting_score * self.WEIGHTS['supporting'] +
            data_quality_score * self.WEIGHTS['data_quality']
        )
        
        self.explanations['dcf'] = {
            'pass': reasons_pass,
            'fail': reasons_fail
        }
        
        if total_score < 0.3:
            self.exclusion_reasons['dcf'] = reasons_fail[0] if reasons_fail else "Insufficient fit criteria"
        
        return round(total_score, 2)
    
    def _score_ddm_single(self) -> float:
        """Calculate fit score for DDM Single-Stage model."""
        reasons_pass = []
        reasons_fail = []
        
        primary_score = 0.0
        supporting_score = 0.0
        data_quality_score = 0.0
        
        # Primary: Has dividends
        if not self.metrics.get('has_dividends', False):
            self.exclusion_reasons['ddm_single_stage'] = "Company does not pay dividends"
            self.explanations['ddm_single_stage'] = {
                'pass': [],
                'fail': ["No dividend payments"]
            }
            return 0.0
        
        primary_score = 1.0
        reasons_pass.append("Company pays dividends")
        
        # Supporting: Stable dividends
        div_stability = self.metrics.get('dividend_stability', 0)
        if div_stability >= 0.7:
            supporting_score += 0.4
            reasons_pass.append(f"Stable dividends (stability: {div_stability:.2f})")
        else:
            supporting_score += div_stability * 0.4
            reasons_fail.append(f"Dividend volatility (stability: {div_stability:.2f})")
        
        # Supporting: Slow dividend growth (single-stage assumes constant growth)
        div_cagr = self.metrics.get('dividend_cagr')
        if div_cagr is not None:
            if div_cagr <= self.THRESHOLDS['ddm_single']['max_dividend_cagr']:
                supporting_score += 0.3
                reasons_pass.append(f"Modest dividend growth ({div_cagr*100:.1f}% CAGR)")
            elif div_cagr < DEFAULT_DISCOUNT_RATE:
                # Growth is higher than ideal but still below discount rate
                supporting_score += 0.15
                reasons_fail.append(
                    f"Dividend growth ({div_cagr*100:.1f}%) is above ideal (<8%) but still usable. "
                    "Consider multi-stage model."
                )
            else:
                # Growth equals or exceeds typical discount rates - CRITICAL ISSUE
                supporting_score += 0.0
                reasons_fail.append(
                    f"Dividend growth ({div_cagr*100:.1f}%) is too high for single-stage model. "
                    "Would produce invalid results. Use multi-stage instead."
                )
        
        # Supporting: Moderate payout ratio
        payout = self.metrics.get('payout_ratio')
        if payout is not None:
            if self.THRESHOLDS['ddm_single']['min_payout_ratio'] <= payout <= self.THRESHOLDS['ddm_single']['max_payout_ratio']:
                supporting_score += 0.3
                reasons_pass.append(f"Sustainable payout ratio ({payout*100:.1f}%)")
            else:
                supporting_score += 0.1
                if payout < self.THRESHOLDS['ddm_single']['min_payout_ratio']:
                    reasons_fail.append(f"Low payout ratio ({payout*100:.1f}%)")
                else:
                    reasons_fail.append(f"High payout ratio ({payout*100:.1f}%) - sustainability concern")
        
        supporting_score = min(1.0, supporting_score)
        
        # Data quality
        if self.metrics['num_years'] >= self.THRESHOLDS['ddm_single']['min_dividend_years']:
            data_quality_score = 1.0
        elif self.metrics['num_years'] >= 3:
            data_quality_score = 0.6
            reasons_fail.append(f"Limited dividend history ({self.metrics['num_years']} years)")
        else:
            data_quality_score = 0.3
            reasons_fail.append(f"Insufficient dividend history ({self.metrics['num_years']} years)")
        
        total_score = (
            primary_score * self.WEIGHTS['primary'] +
            supporting_score * self.WEIGHTS['supporting'] +
            data_quality_score * self.WEIGHTS['data_quality']
        )
        
        self.explanations['ddm_single_stage'] = {
            'pass': reasons_pass,
            'fail': reasons_fail
        }
        
        if total_score < 0.3:
            self.exclusion_reasons['ddm_single_stage'] = reasons_fail[0] if reasons_fail else "Insufficient fit criteria"
        
        return round(total_score, 2)
    
    def _score_ddm_multi(self) -> float:
        """Calculate fit score for DDM Multi-Stage model."""
        reasons_pass = []
        reasons_fail = []
        
        primary_score = 0.0
        supporting_score = 0.0
        data_quality_score = 0.0
        
        # Primary: Has dividends
        if not self.metrics.get('has_dividends', False):
            self.exclusion_reasons['ddm_multi_stage'] = "Company does not pay dividends"
            self.explanations['ddm_multi_stage'] = {
                'pass': [],
                'fail': ["No dividend payments"]
            }
            return 0.0
        
        primary_score = 1.0
        reasons_pass.append("Company pays dividends")
        
        # Supporting: Growing dividends
        div_cagr = self.metrics.get('dividend_cagr')
        if div_cagr is not None:
            if div_cagr >= self.THRESHOLDS['ddm_multi']['min_dividend_cagr']:
                supporting_score += 0.5
                reasons_pass.append(f"Strong dividend growth ({div_cagr*100:.1f}% CAGR)")
            elif div_cagr > 0:
                supporting_score += 0.2
                reasons_pass.append(f"Positive dividend growth ({div_cagr*100:.1f}% CAGR)")
            else:
                reasons_fail.append(f"Declining dividends ({div_cagr*100:.1f}% CAGR)")
        
        # Supporting: EPS growth (supports dividend growth)
        eps_cagr = self.metrics.get('eps_cagr')
        if eps_cagr is not None and eps_cagr > self.THRESHOLDS['ddm_multi']['min_eps_growth']:
            supporting_score += 0.5
            reasons_pass.append(f"Positive EPS growth ({eps_cagr*100:.1f}% CAGR)")
        elif eps_cagr is not None:
            supporting_score += 0.1
            reasons_fail.append(f"Weak EPS growth ({eps_cagr*100:.1f}% CAGR)")
        
        supporting_score = min(1.0, supporting_score)
        
        # Data quality
        if self.metrics['num_years'] >= self.THRESHOLDS['ddm_multi']['min_dividend_years']:
            data_quality_score = 1.0
        else:
            data_quality_score = 0.5
            reasons_fail.append(f"Limited history ({self.metrics['num_years']} years)")
        
        total_score = (
            primary_score * self.WEIGHTS['primary'] +
            supporting_score * self.WEIGHTS['supporting'] +
            data_quality_score * self.WEIGHTS['data_quality']
        )
        
        self.explanations['ddm_multi_stage'] = {
            'pass': reasons_pass,
            'fail': reasons_fail
        }
        
        if total_score < 0.3:
            self.exclusion_reasons['ddm_multi_stage'] = reasons_fail[0] if reasons_fail else "Insufficient fit criteria"
        
        return round(total_score, 2)
    
    def _score_pe_model(self) -> float:
        """Calculate fit score for P/E Model."""
        reasons_pass = []
        reasons_fail = []
        
        primary_score = 0.0
        supporting_score = 0.0
        data_quality_score = 0.0
        
        # Primary: Positive earnings
        eps_positive_years = self.metrics.get('eps_positive_years', 0)
        
        if eps_positive_years >= self.THRESHOLDS['pe_model']['min_eps_positive_years']:
            primary_score = min(1.0, eps_positive_years / self.metrics['num_years'])
            reasons_pass.append(f"Positive earnings in {eps_positive_years}/{self.metrics['num_years']} years")
        else:
            reasons_fail.append(f"Insufficient positive earnings ({eps_positive_years} years)")
        
        # Supporting: EPS stability
        eps_stability = self.metrics.get('eps_stability', 0)
        if eps_stability >= 0.5:
            supporting_score += 0.5
            reasons_pass.append(f"Stable earnings (stability: {eps_stability:.2f})")
        else:
            supporting_score += eps_stability * 0.5
            reasons_fail.append(f"Volatile earnings (stability: {eps_stability:.2f})")
        
        # Supporting: Reasonable P/E range
        avg_pe = self.metrics.get('avg_pe')
        if avg_pe is not None:
            if self.THRESHOLDS['pe_model']['min_pe'] <= avg_pe <= self.THRESHOLDS['pe_model']['max_pe']:
                supporting_score += 0.5
                reasons_pass.append(f"Reasonable P/E range (avg: {avg_pe:.1f})")
            else:
                supporting_score += 0.2
                reasons_fail.append(f"Extreme P/E values (avg: {avg_pe:.1f})")
        
        supporting_score = min(1.0, supporting_score)
        
        # Data quality: P/E volatility
        pe_volatility = self.metrics.get('pe_volatility')
        if pe_volatility is not None:
            if pe_volatility <= 0.5:
                data_quality_score = 1.0
            elif pe_volatility <= self.THRESHOLDS['pe_model']['max_pe_volatility']:
                data_quality_score = 0.7
                reasons_fail.append(f"Moderate P/E volatility ({pe_volatility:.2f})")
            else:
                data_quality_score = 0.4
                reasons_fail.append(f"High P/E volatility ({pe_volatility:.2f})")
        else:
            data_quality_score = 0.5
        
        total_score = (
            primary_score * self.WEIGHTS['primary'] +
            supporting_score * self.WEIGHTS['supporting'] +
            data_quality_score * self.WEIGHTS['data_quality']
        )
        
        self.explanations['pe_model'] = {
            'pass': reasons_pass,
            'fail': reasons_fail
        }
        
        if total_score < 0.3:
            self.exclusion_reasons['pe_model'] = reasons_fail[0] if reasons_fail else "Insufficient fit criteria"
        
        return round(total_score, 2)
    
    def _score_graham(self) -> float:
        """Calculate fit score for Graham Formula."""
        reasons_pass = []
        reasons_fail = []
        
        primary_score = 0.0
        supporting_score = 0.0
        data_quality_score = 0.0
        
        # Primary: Consistent positive earnings
        eps_positive_years = self.metrics.get('eps_positive_years', 0)
        
        if eps_positive_years >= self.THRESHOLDS['graham']['min_eps_years']:
            primary_score = 1.0
            reasons_pass.append(f"Consistent profitability ({eps_positive_years} years)")
        else:
            primary_score = eps_positive_years / self.THRESHOLDS['graham']['min_eps_years']
            reasons_fail.append(f"Inconsistent earnings ({eps_positive_years}/{self.THRESHOLDS['graham']['min_eps_years']} years)")
        
        # Supporting: Strong ROE
        roe = self.metrics.get('roe')
        if roe is not None:
            if roe >= self.THRESHOLDS['graham']['min_roe']:
                supporting_score += 0.5
                reasons_pass.append(f"Strong ROE ({roe*100:.1f}%)")
            else:
                supporting_score += max(0, roe / self.THRESHOLDS['graham']['min_roe']) * 0.5
                reasons_fail.append(f"Weak ROE ({roe*100:.1f}%)")
        
        # Supporting: Low debt
        debt_equity = self.metrics.get('debt_equity')
        if debt_equity is not None:
            if debt_equity <= self.THRESHOLDS['graham']['max_debt_equity']:
                supporting_score += 0.5
                reasons_pass.append(f"Conservative debt levels (D/E: {debt_equity:.2f})")
            else:
                supporting_score += max(0, 1 - (debt_equity / self.THRESHOLDS['graham']['max_debt_equity'])) * 0.5
                reasons_fail.append(f"High debt (D/E: {debt_equity:.2f})")
        
        supporting_score = min(1.0, supporting_score)
        
        # Data quality
        if self.metrics['num_years'] >= self.THRESHOLDS['graham']['min_eps_years']:
            data_quality_score = 1.0
        else:
            data_quality_score = 0.5
            reasons_fail.append(f"Limited history ({self.metrics['num_years']} years)")
        
        total_score = (
            primary_score * self.WEIGHTS['primary'] +
            supporting_score * self.WEIGHTS['supporting'] +
            data_quality_score * self.WEIGHTS['data_quality']
        )
        
        self.explanations['graham_value'] = {
            'pass': reasons_pass,
            'fail': reasons_fail
        }
        
        if total_score < 0.3:
            self.exclusion_reasons['graham_value'] = reasons_fail[0] if reasons_fail else "Insufficient fit criteria"
        
        return round(total_score, 2)
    
    def _score_asset_based(self) -> float:
        """Calculate fit score for Asset-Based valuation."""
        reasons_pass = []
        reasons_fail = []
        
        primary_score = 0.0
        supporting_score = 0.0
        data_quality_score = 0.0
        
        # Primary: Positive tangible book value
        tangible_book = self.metrics.get('tangible_book_value')
        asset_quality = self.metrics.get('asset_quality')
        
        if tangible_book is not None and tangible_book > self.THRESHOLDS['asset_based']['min_tangible_book']:
            primary_score = 0.5
            reasons_pass.append(f"Positive book value (${tangible_book:.2f}/share)")
        else:
            reasons_fail.append("Negative or zero book value")
        
        if asset_quality is not None and asset_quality >= self.THRESHOLDS['asset_based']['min_asset_quality']:
            primary_score += 0.5
            reasons_pass.append(f"Good asset quality ({asset_quality*100:.1f}%)")
        else:
            reasons_fail.append(f"Poor asset quality ({asset_quality*100:.1f}%)" if asset_quality else "Cannot assess asset quality")
        
        # Supporting: Low profitability (makes asset-based more relevant)
        net_margin = self.metrics.get('net_margin')
        if net_margin is not None:
            if net_margin < self.THRESHOLDS['asset_based']['use_when_margin_below']:
                supporting_score += 0.5
                reasons_pass.append(f"Low margins ({net_margin*100:.1f}%) - assets more relevant")
            else:
                supporting_score += 0.2
                reasons_fail.append(f"Profitable company ({net_margin*100:.1f}%) - earnings-based models may be better")
        
        # Supporting: Negative or volatile earnings
        eps_stability = self.metrics.get('eps_stability', 0)
        if eps_stability < 0.5:
            supporting_score += 0.5
            reasons_pass.append(f"Volatile earnings (stability: {eps_stability:.2f}) - assets provide floor value")
        else:
            supporting_score += 0.2
        
        supporting_score = min(1.0, supporting_score)
        
        # Data quality: Balance sheet availability
        if asset_quality is not None and tangible_book is not None:
            data_quality_score = 1.0
        else:
            data_quality_score = 0.3
            reasons_fail.append("Incomplete balance sheet data")
        
        total_score = (
            primary_score * self.WEIGHTS['primary'] +
            supporting_score * self.WEIGHTS['supporting'] +
            data_quality_score * self.WEIGHTS['data_quality']
        )
        
        self.explanations['asset_based'] = {
            'pass': reasons_pass,
            'fail': reasons_fail
        }
        
        if total_score < 0.3:
            self.exclusion_reasons['asset_based'] = reasons_fail[0] if reasons_fail else "Insufficient fit criteria"
        
        return round(total_score, 2)
    
    def get_recommended_models(self, min_score: float = 0.5) -> List[str]:
        """
        Get list of recommended models based on fit scores.
        
        Args:
            min_score: Minimum fit score to recommend a model (default 0.5)
            
        Returns:
            List of recommended model names
        """
        if not self.fit_scores:
            self.calculate_fit_scores()
        
        recommended = [
            model for model, score in self.fit_scores.items()
            if score >= min_score
        ]
        
        # Sort by score (highest first)
        recommended.sort(key=lambda m: self.fit_scores[m], reverse=True)
        
        logger.info(f"Recommended models (min_score={min_score}): {recommended}")
        
        return recommended
    
    def get_exclusion_reasons(self) -> Dict[str, str]:
        """
        Get reasons why models were excluded.
        
        Returns:
            Dictionary with model names and exclusion reasons
        """
        if not self.fit_scores:
            self.calculate_fit_scores()
        
        return self.exclusion_reasons.copy()
    
    def get_fit_explanations(self, model: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Get detailed explanations for fit scores.
        
        Args:
            model: Specific model to get explanation for (optional)
            
        Returns:
            Dictionary with pass/fail reasons for each model
        """
        if not self.fit_scores:
            self.calculate_fit_scores()
        
        if model:
            return {model: self.explanations.get(model, {'pass': [], 'fail': []})}
        
        return self.explanations.copy()
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of model selection analysis.
        
        Returns:
            Dictionary with all selection information
        """
        if not self.fit_scores:
            self.calculate_fit_scores()
        
        recommended = self.get_recommended_models(min_score=0.5)
        
        summary = {
            'fit_scores': self.fit_scores.copy(),
            'recommended_models': recommended,
            'excluded_models': list(self.exclusion_reasons.keys()),
            'explanations': self.explanations.copy(),
            'exclusion_reasons': self.exclusion_reasons.copy(),
            'metrics': self.metrics.copy(),
            'top_model': recommended[0] if recommended else None,
            'top_score': self.fit_scores[recommended[0]] if recommended else 0.0
        }
        
        return summary
    
    def print_analysis(self, detailed: bool = False) -> None:
        """
        Print formatted analysis of model selection.
        
        Args:
            detailed: If True, show detailed explanations
        """
        if not self.fit_scores:
            self.calculate_fit_scores()
        
        print("\n" + "="*70)
        print("MODEL SELECTION ANALYSIS")
        print("="*70)
        
        # Sort models by fit score
        sorted_models = sorted(self.fit_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFit Scores (0.0 - 1.0):")
        print("-" * 70)
        
        for model, score in sorted_models:
            model_name = MODEL_DISPLAY_NAMES.get(model, model)
            
            # Visual bar
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Status
            if score >= 0.7:
                status = "✅ HIGHLY RECOMMENDED"
            elif score >= 0.5:
                status = "✓ RECOMMENDED"
            elif score >= 0.3:
                status = "⚠ MARGINAL"
            else:
                status = "❌ NOT RECOMMENDED"
            
            print(f"{model_name:30s} {bar} {score:.2f} {status}")
        
        # Show recommendations
        recommended = self.get_recommended_models(min_score=0.5)
        
        print("\n" + "="*70)
        print(f"RECOMMENDED MODELS: {len(recommended)}")
        print("="*70)
        
        if recommended:
            for model in recommended:
                model_name = MODEL_DISPLAY_NAMES.get(model, model)
                score = self.fit_scores[model]
                print(f"\n✅ {model_name} (Fit: {score:.2f})")
                
                if detailed and model in self.explanations:
                    explanations = self.explanations[model]
                    
                    if explanations['pass']:
                        print("   Strengths:")
                        for reason in explanations['pass']:
                            print(f"     ✓ {reason}")
                    
                    if explanations['fail']:
                        print("   Concerns:")
                        for reason in explanations['fail']:
                            print(f"     ⚠ {reason}")
        else:
            print("\n⚠️ No models meet the minimum fit criteria.")
            print("   Consider reviewing the company's financial data quality.")
        
        # Show excluded models
        if self.exclusion_reasons:
            print("\n" + "="*70)
            print("EXCLUDED MODELS")
            print("="*70)
            
            for model, reason in self.exclusion_reasons.items():
                model_name = MODEL_DISPLAY_NAMES.get(model, model)
                score = self.fit_scores.get(model, 0.0)
                print(f"\n❌ {model_name} (Fit: {score:.2f})")
                print(f"   Reason: {reason}")
                
                if detailed and model in self.explanations:
                    explanations = self.explanations[model]
                    if explanations['fail']:
                        print("   Details:")
                        for detail in explanations['fail']:
                            print(f"     • {detail}")
        
        print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    # This would normally use real data from IVSimplifier
    # Here's a demonstration of how to use it
    
    print("ModelSelector - Intelligent Valuation Model Selection")
    print("\nThis module analyzes company financials and recommends")
    print("the most appropriate valuation models based on:")
    print("  • Cash flow characteristics")
    print("  • Dividend policy")
    print("  • Earnings stability")
    print("  • Balance sheet strength")
    print("  • Data availability and quality")
    print("\nIntegrate with your existing pipeline:")
    print("  1. Fetch data with DataFetcher")
    print("  2. Simplify with IVSimplifier")
    print("  3. Select models with ModelSelector")
    print("  4. Calculate valuations with ValuationCalculator")