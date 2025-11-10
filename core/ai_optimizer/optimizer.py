import logging
from typing import Dict, Optional
import json
from core.ai_provider.client import UnifiedAIClient
from core.ai_provider.enums import AIProvider
from core.ai_optimizer.wacc import calculate_wacc
from .prompts import build_ai_prompt, build_rule_based_explanation

logger = logging.getLogger(__name__)

class AIParameterOptimizer:
    """Optimize DCF parameters using AI or rule-based methods."""

    INDUSTRY_DISCOUNT_RATES = {
        'Technology': 0.12, 'Communication Services': 0.11,
        'Consumer Cyclical': 0.11, 'Consumer Defensive': 0.08,
        'Healthcare': 0.09, 'Financial Services': 0.11,
        'Financials': 0.11, 'Basic Materials': 0.10,
        'Energy': 0.10, 'Industrials': 0.10,
        'Real Estate': 0.09, 'Utilities': 0.08,
        'default': 0.10
    }

    TERMINAL_GROWTH_RATES = {
        'mature': 0.025, 'developing': 0.035,
        'high_growth': 0.04, 'declining': 0.015,
        'default': 0.025
    }

    def __init__(self, api_key: Optional[str] = None, use_ollama: bool = False, use_ai: bool = True):
        self.api_key = api_key
        self.use_ollama = use_ollama
        self.use_ai = use_ai and (api_key or use_ollama)

        if self.use_ai:
            provider = AIProvider.OLLAMA if use_ollama else AIProvider.GROQ if api_key else AIProvider.NONE
            self.ai_client = UnifiedAIClient(provider, api_key)

            if self.ai_client.is_available:
                logger.info(f"AI Parameter Optimizer initialized: {self.ai_client.get_provider_name()}")
            else:
                logger.warning("AI unavailable, using rule-based mode")
                self.use_ai = False
        else:
            self.ai_client = UnifiedAIClient(AIProvider.NONE)
            logger.info("AI Parameter Optimizer in rule-based mode")

    def optimize_parameters(self, ticker: str, company_data: Dict, financial_metrics: Dict) -> Dict[str, any]:
        if self.use_ai:
            return self._optimize_with_ai(ticker, company_data, financial_metrics)
        else:
            return self._optimize_rule_based(ticker, company_data, financial_metrics)

    def _optimize_rule_based(self, ticker: str, company_data: Dict, financial_metrics: Dict) -> Dict[str, any]:
        sector = company_data.get('sector', 'default')
        market_cap = company_data.get('marketCap', 0)
        beta = financial_metrics.get('beta', 1.0)
        debt_to_equity = financial_metrics.get('debt_to_equity', 0)
        revenue_growth = financial_metrics.get('revenue_cagr', 0)

        # 1. Discount Rate
        base_rate = self.INDUSTRY_DISCOUNT_RATES.get(sector, 0.10)
        size_adj = 0.03 if market_cap < 2e9 else 0.01 if market_cap < 10e9 else 0.0
        beta_adj = (beta - 1.0) * 0.02
        leverage_adj = 0.02 if debt_to_equity > 1.5 else 0.01 if debt_to_equity > 0.8 else 0.0
        discount_rate = max(0.06, min(0.20, base_rate + size_adj + beta_adj + leverage_adj))

        # 2. Terminal Growth
        if revenue_growth > 0.15:
            terminal_growth = self.TERMINAL_GROWTH_RATES['high_growth']; market_phase='high_growth'
        elif revenue_growth > 0.05:
            terminal_growth = self.TERMINAL_GROWTH_RATES['mature']; market_phase='mature'
        elif revenue_growth < 0:
            terminal_growth = self.TERMINAL_GROWTH_RATES['declining']; market_phase='declining'
        else:
            terminal_growth = self.TERMINAL_GROWTH_RATES['default']; market_phase='mature'

        # 3. Projection Years
        if revenue_growth > 0.2: projection_years=7
        elif revenue_growth > 0.1: projection_years=6
        else: projection_years=5

        explanation = build_rule_based_explanation(
            ticker, sector, discount_rate, terminal_growth, base_rate, size_adj, beta_adj,
            leverage_adj, market_cap, beta, debt_to_equity, revenue_growth, market_phase
        )

        return {
            'ticker': ticker,
            'discount_rate': round(discount_rate, 4),
            'terminal_growth': round(terminal_growth, 4),
            'projection_years': projection_years,
            'confidence': 'Medium',
            'method': 'Rule-Based',
            'explanation': explanation,
            'adjustments': {
                'base_rate': base_rate,
                'size_adjustment': size_adj,
                'beta_adjustment': beta_adj,
                'leverage_adjustment': leverage_adj
            }
        }

    def _optimize_with_ai(self, ticker: str, company_data: Dict, financial_metrics: Dict) -> Dict[str, any]:
        rule_based = self._optimize_rule_based(ticker, company_data, financial_metrics)
        prompt = build_ai_prompt(ticker, company_data, financial_metrics, rule_based)
        ai_response = self.ai_client.chat(
            messages=[
                {"role": "system", "content": "You are an expert financial analyst specializing in DCF valuation. Respond ONLY with JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        if not ai_response:
            logger.warning("AI returned no response; falling back to rule-based")
            return rule_based

        # Extract JSON from possible markdown
        if '```json' in ai_response:
            ai_response = ai_response.split('```json')[1].split('```')[0].strip()
        elif '```' in ai_response:
            ai_response = ai_response.split('```')[1].split('```')[0].strip()

        try:
            ai_params = json.loads(ai_response)
        except Exception as e:
            logger.error(f"Failed to parse AI JSON: {e}")
            return rule_based

        # Apply sanity checks
        discount_rate = max(0.05, min(0.25, float(ai_params.get('discount_rate', rule_based['discount_rate']))))
        terminal_growth = max(0.01, min(0.05, float(ai_params.get('terminal_growth', rule_based['terminal_growth']))))
        projection_years = max(3, min(10, int(ai_params.get('projection_years', rule_based['projection_years']))))
        if discount_rate <= terminal_growth:
            discount_rate = terminal_growth + 0.03

        return {
            'ticker': ticker,
            'discount_rate': round(discount_rate, 4),
            'terminal_growth': round(terminal_growth, 4),
            'projection_years': projection_years,
            'confidence': 'High',
            'method': 'AI-Powered',
            'explanation': ai_params.get('explanation', 'AI-optimized parameters'),
            'ai_reasoning': ai_params.get('reasoning', ''),
            'rule_based_baseline': rule_based
        }

    def batch_optimize(self, companies: Dict[str, tuple]) -> Dict[str, Dict]:
        results = {}
        for ticker, (company_data, financial_metrics) in companies.items():
            try:
                results[ticker] = self.optimize_parameters(ticker, company_data, financial_metrics)
            except Exception as e:
                logger.error(f"Failed to optimize {ticker}: {e}")
                results[ticker] = None
        return results
