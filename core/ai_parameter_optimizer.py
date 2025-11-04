# ai_parameter_optimizer.py
"""
AI-Powered Parameter Optimization
Intelligently determines optimal DCF parameters based on company characteristics
"""

import logging
from typing import Dict, Optional, Tuple
from groq import Groq
from core.ai_provider import UnifiedAIClient, AIProvider
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIParameterOptimizer:
    """
    Uses AI and rule-based logic to determine optimal valuation parameters.
    
    Determines:
    - Company-specific discount rate (WACC)
    - Terminal growth rate
    - Projection years
    - Growth rate assumptions
    
    Methods:
    1. Rule-based: Fast, deterministic, based on industry/financial metrics
    2. AI-powered: Contextual, considers multiple factors, uses Groq API
    """
    
    # Industry-specific discount rates (based on typical industry risk profiles)
    INDUSTRY_DISCOUNT_RATES = {
        'Technology': 0.12,
        'Communication Services': 0.11,
        'Consumer Cyclical': 0.11,
        'Consumer Defensive': 0.08,
        'Healthcare': 0.09,
        'Financial Services': 0.11,
        'Financials': 0.11,
        'Basic Materials': 0.10,
        'Energy': 0.10,
        'Industrials': 0.10,
        'Real Estate': 0.09,
        'Utilities': 0.08,
        'default': 0.10
    }
    
    # Terminal growth rates by market maturity
    TERMINAL_GROWTH_RATES = {
        'mature': 0.025,      # 2.5% - Developed markets
        'developing': 0.035,  # 3.5% - Emerging markets
        'high_growth': 0.04,  # 4.0% - High growth sectors
        'declining': 0.015,   # 1.5% - Declining industries
        'default': 0.025
    }
    
    def __init__(self, api_key: Optional[str] = None, use_ollama: bool = False, use_ai: bool = True):
        """Initialize optimizer with Groq or Ollama support."""
        self.api_key = api_key
        self.use_ollama = use_ollama
        self.use_ai = use_ai and (api_key or use_ollama)
        
        if self.use_ai:
            # Determine provider
            provider = AIProvider.OLLAMA if use_ollama else AIProvider.GROQ if api_key else AIProvider.NONE
            self.ai_client = UnifiedAIClient(provider, api_key)
            
            if self.ai_client.is_available:
                logger.info(f"AI Parameter Optimizer initialized: {self.ai_client.get_provider_name()}")
            else:
                logger.warning("AI unavailable, using rule-based mode")
                self.use_ai = False
            
            # Backward compatibility
            self.client = self.ai_client if self.ai_client.is_available else None
        else:
            self.ai_client = UnifiedAIClient(AIProvider.NONE)
            self.client = None
            logger.info("AI Parameter Optimizer in rule-based mode")
    
    def optimize_parameters(
        self,
        ticker: str,
        company_data: Dict,
        financial_metrics: Dict
    ) -> Dict[str, any]:
        """
        Determine optimal parameters for a company.
        
        Args:
            ticker: Stock ticker symbol
            company_data: Company info (sector, market cap, etc.)
            financial_metrics: Financial metrics (beta, debt, growth rates, etc.)
            
        Returns:
            Dictionary with optimized parameters and explanations
        """
        if self.use_ai:
            return self._optimize_with_ai(ticker, company_data, financial_metrics)
        else:
            return self._optimize_rule_based(ticker, company_data, financial_metrics)
    
    def _optimize_rule_based(
        self,
        ticker: str,
        company_data: Dict,
        financial_metrics: Dict
    ) -> Dict[str, any]:
        """
        Rule-based parameter optimization.
        Fast and deterministic.
        """
        logger.info(f"Optimizing parameters for {ticker} (rule-based)")
        
        # Extract company characteristics
        sector = company_data.get('sector', 'default')
        market_cap = company_data.get('marketCap', 0)
        beta = financial_metrics.get('beta', 1.0)
        debt_to_equity = financial_metrics.get('debt_to_equity', 0)
        revenue_growth = financial_metrics.get('revenue_cagr', 0)
        
        # 1. Calculate Discount Rate
        base_rate = self.INDUSTRY_DISCOUNT_RATES.get(sector, 0.10)
        
        # Adjust for company size (small cap = higher risk)
        if market_cap < 2_000_000_000:  # < $2B
            size_adjustment = 0.03
        elif market_cap < 10_000_000_000:  # < $10B
            size_adjustment = 0.01
        else:
            size_adjustment = 0.0
        
        # Adjust for beta (systematic risk)
        beta_adjustment = (beta - 1.0) * 0.02  # +/- 2% per beta point
        
        # Adjust for leverage (financial risk)
        if debt_to_equity > 1.5:
            leverage_adjustment = 0.02
        elif debt_to_equity > 0.8:
            leverage_adjustment = 0.01
        else:
            leverage_adjustment = 0.0
        
        discount_rate = base_rate + size_adjustment + beta_adjustment + leverage_adjustment
        discount_rate = max(0.06, min(0.20, discount_rate))  # Clamp between 6-20%
        
        # 2. Determine Terminal Growth Rate
        if revenue_growth > 0.15:  # High growth
            terminal_growth = self.TERMINAL_GROWTH_RATES['high_growth']
            market_phase = 'high_growth'
        elif revenue_growth > 0.05:  # Moderate growth
            terminal_growth = self.TERMINAL_GROWTH_RATES['mature']
            market_phase = 'mature'
        elif revenue_growth < 0:  # Declining
            terminal_growth = self.TERMINAL_GROWTH_RATES['declining']
            market_phase = 'declining'
        else:
            terminal_growth = self.TERMINAL_GROWTH_RATES['default']
            market_phase = 'mature'
        
        # 3. Determine Projection Years
        if revenue_growth > 0.20:  # Very high growth
            projection_years = 7
        elif revenue_growth > 0.10:  # High growth
            projection_years = 6
        else:  # Stable/mature
            projection_years = 5
        
        # 4. Build explanation
        explanation = self._build_rule_based_explanation(
            ticker, sector, discount_rate, terminal_growth,
            base_rate, size_adjustment, beta_adjustment, leverage_adjustment,
            market_cap, beta, debt_to_equity, revenue_growth, market_phase
        )
        
        result = {
            'ticker': ticker,
            'discount_rate': round(discount_rate, 4),
            'terminal_growth': round(terminal_growth, 4),
            'projection_years': projection_years,
            'confidence': 'Medium',  # Rule-based has medium confidence
            'method': 'Rule-Based',
            'explanation': explanation,
            'adjustments': {
                'base_rate': base_rate,
                'size_adjustment': size_adjustment,
                'beta_adjustment': beta_adjustment,
                'leverage_adjustment': leverage_adjustment
            }
        }
        
        logger.info(f"{ticker}: Discount Rate = {discount_rate:.2%}, Terminal Growth = {terminal_growth:.2%}")
        
        return result
    
    def _optimize_with_ai(
        self,
        ticker: str,
        company_data: Dict,
        financial_metrics: Dict
    ) -> Dict[str, any]:
        """
        AI-powered parameter optimization.
        Uses Groq API for contextual reasoning.
        """
        logger.info(f"Optimizing parameters for {ticker} (AI-powered)")
        
        try:
            # First, get rule-based parameters as baseline
            rule_based = self._optimize_rule_based(ticker, company_data, financial_metrics)
            
            # Build prompt for AI
            prompt = self._build_ai_prompt(ticker, company_data, financial_metrics, rule_based)
            
            # Call Groq API
            ai_response = self.ai_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial analyst specializing in DCF valuation. 
            Your task is to determine optimal discount rates and terminal growth rates for companies based on their characteristics.
            Consider industry dynamics, company-specific risks, growth trajectory, and financial health.
            Respond ONLY with valid JSON, no additional text."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800
            )

            if not ai_response:
                logger.error("AI returned no response")
                return self._optimize_rule_based(ticker, company_data, financial_metrics)
            
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                ai_response = ai_response.split('```')[1].split('```')[0].strip()
            
            ai_params = json.loads(ai_response)
            
            # Validate AI recommendations
            discount_rate = float(ai_params.get('discount_rate', rule_based['discount_rate']))
            terminal_growth = float(ai_params.get('terminal_growth', rule_based['terminal_growth']))
            projection_years = int(ai_params.get('projection_years', rule_based['projection_years']))
            
            # Sanity checks
            discount_rate = max(0.05, min(0.25, discount_rate))
            terminal_growth = max(0.01, min(0.05, terminal_growth))
            projection_years = max(3, min(10, projection_years))
            
            # Ensure discount rate > terminal growth
            if discount_rate <= terminal_growth:
                discount_rate = terminal_growth + 0.03
            
            result = {
                'ticker': ticker,
                'discount_rate': round(discount_rate, 4),
                'terminal_growth': round(terminal_growth, 4),
                'projection_years': projection_years,
                'confidence': 'High',  # AI has high confidence
                'method': 'AI-Powered',
                'explanation': ai_params.get('explanation', 'AI-optimized parameters'),
                'ai_reasoning': ai_params.get('reasoning', ''),
                'rule_based_baseline': rule_based
            }
            
            logger.info(f"{ticker} (AI): Discount Rate = {discount_rate:.2%}, Terminal Growth = {terminal_growth:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI optimization failed for {ticker}: {e}")
            logger.info(f"Falling back to rule-based for {ticker}")
            # Fallback to rule-based
            return self._optimize_rule_based(ticker, company_data, financial_metrics)
    
    def _build_ai_prompt(
        self,
        ticker: str,
        company_data: Dict,
        financial_metrics: Dict,
        rule_based: Dict
    ) -> str:
        """Build prompt for AI reasoning."""
        
        prompt = f"""Analyze this company and determine optimal DCF parameters:

COMPANY: {ticker}
Sector: {company_data.get('sector', 'Unknown')}
Industry: {company_data.get('industry', 'Unknown')}
Market Cap: ${company_data.get('marketCap', 0):,.0f}

FINANCIAL METRICS:
- Beta: {financial_metrics.get('beta', 'N/A')}
- Debt/Equity: {financial_metrics.get('debt_to_equity', 'N/A')}
- Revenue Growth (CAGR): {financial_metrics.get('revenue_cagr', 'N/A'):.1%} if isinstance(financial_metrics.get('revenue_cagr'), (int, float)) else 'N/A'
- FCF Growth (CAGR): {financial_metrics.get('fcf_cagr', 'N/A'):.1%} if isinstance(financial_metrics.get('fcf_cagr'), (int, float)) else 'N/A'
- ROE: {financial_metrics.get('roe', 'N/A'):.1%} if isinstance(financial_metrics.get('roe'), (int, float)) else 'N/A'
- Net Margin: {financial_metrics.get('net_margin', 'N/A'):.1%} if isinstance(financial_metrics.get('net_margin'), (int, float)) else 'N/A'

RULE-BASED BASELINE:
- Discount Rate: {rule_based['discount_rate']:.2%}
- Terminal Growth: {rule_based['terminal_growth']:.2%}
- Projection Years: {rule_based['projection_years']}

TASK:
Determine optimal parameters considering:
1. Industry risk and cyclicality
2. Company-specific risk factors
3. Growth trajectory and sustainability
4. Financial health and leverage
5. Competitive position

Respond with JSON in this exact format:
{{
  "discount_rate": 0.10,
  "terminal_growth": 0.025,
  "projection_years": 5,
  "explanation": "Brief explanation of parameter choices (2-3 sentences)",
  "reasoning": "Detailed reasoning covering key factors considered"
}}

CONSTRAINTS:
- Discount rate: 5% to 25%
- Terminal growth: 1% to 5%
- Projection years: 3 to 10
- Discount rate MUST be > terminal growth rate
"""
        
        return prompt
    
    def _build_rule_based_explanation(
        self,
        ticker: str,
        sector: str,
        discount_rate: float,
        terminal_growth: float,
        base_rate: float,
        size_adj: float,
        beta_adj: float,
        leverage_adj: float,
        market_cap: float,
        beta: float,
        debt_to_equity: float,
        revenue_growth: float,
        market_phase: str
    ) -> str:
        """Build human-readable explanation for rule-based parameters."""
        
        size_category = "large-cap" if market_cap > 10_000_000_000 else "mid-cap" if market_cap > 2_000_000_000 else "small-cap"
        
        explanation = f"""**{ticker} Parameter Analysis:**

**Discount Rate: {discount_rate:.2%}**
- Base rate for {sector}: {base_rate:.2%}
- Size adjustment ({size_category}): +{size_adj:.2%}
- Beta risk (β={beta:.2f}): {beta_adj:+.2%}
- Leverage risk (D/E={debt_to_equity:.2f}): +{leverage_adj:.2%}

**Terminal Growth: {terminal_growth:.2%}**
- Market phase: {market_phase.replace('_', ' ').title()}
- Revenue growth: {revenue_growth:.1%}

**Rationale:**
"""
        
        # Add specific rationale
        if size_adj > 0:
            explanation += f"\n- Higher risk premium for {size_category} company"
        if beta > 1.2:
            explanation += f"\n- Elevated systematic risk (high beta)"
        elif beta < 0.8:
            explanation += f"\n- Lower systematic risk (low beta)"
        if leverage_adj > 0:
            explanation += f"\n- Additional risk for high financial leverage"
        if revenue_growth > 0.15:
            explanation += f"\n- Strong growth justifies longer projection period"
        
        return explanation
    
    def batch_optimize(
        self,
        companies: Dict[str, Tuple[Dict, Dict]]
    ) -> Dict[str, Dict]:
        """
        Optimize parameters for multiple companies.
        
        Args:
            companies: Dict of {ticker: (company_data, financial_metrics)}
            
        Returns:
            Dict of {ticker: optimization_results}
        """
        results = {}
        
        for ticker, (company_data, financial_metrics) in companies.items():
            try:
                results[ticker] = self.optimize_parameters(ticker, company_data, financial_metrics)
            except Exception as e:
                logger.error(f"Failed to optimize {ticker}: {e}")
                results[ticker] = None
        
        return results


def calculate_wacc(company_data: Dict, financial_metrics: Dict) -> float:
    """
    Calculate company-specific WACC (Weighted Average Cost of Capital).
    
    WACC = (E/V × Re) + (D/V × Rd × (1-Tc))
    Where:
    - E = Market value of equity
    - D = Market value of debt
    - V = E + D
    - Re = Cost of equity
    - Rd = Cost of debt
    - Tc = Corporate tax rate
    """
    try:
        # Get values
        market_cap = company_data.get('marketCap', 0)
        total_debt = financial_metrics.get('total_debt', 0)
        beta = financial_metrics.get('beta', 1.0)
        
        # Cost of equity (CAPM: Rf + β(Rm - Rf))
        risk_free_rate = 0.04  # 10-year Treasury yield
        market_return = 0.10
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        
        # If no debt, WACC = cost of equity
        if total_debt == 0 or market_cap == 0:
            return cost_of_equity
        
        # Total value
        total_value = market_cap + total_debt
        
        # Weights
        equity_weight = market_cap / total_value
        debt_weight = total_debt / total_value
        
        # Cost of debt (approximate)
        interest_expense = financial_metrics.get('interest_expense', 0)
        cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.05
        
        # Tax rate
        tax_rate = financial_metrics.get('tax_rate', 0.21)  # Federal corporate rate
        
        # WACC
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        
        return max(0.05, min(0.25, wacc))  # Clamp to reasonable range
        
    except Exception as e:
        logger.error(f"WACC calculation failed: {e}")
        return 0.10  # Default fallback


# Example usage
if __name__ == "__main__":
    import os
    
    print("="*70)
    print("AI PARAMETER OPTIMIZER TEST")
    print("="*70)
    
    # Sample company data
    sample_company = {
        'sector': 'Technology',
        'industry': 'Software',
        'marketCap': 2_500_000_000_000  # $2.5T
    }
    
    sample_metrics = {
        'beta': 1.2,
        'debt_to_equity': 0.3,
        'revenue_cagr': 0.15,
        'fcf_cagr': 0.18,
        'roe': 0.35,
        'net_margin': 0.25
    }
    
    # Test rule-based
    print("\n[1/2] Testing Rule-Based Optimization...")
    optimizer_rule = AIParameterOptimizer(use_ai=False)
    result_rule = optimizer_rule.optimize_parameters('AAPL', sample_company, sample_metrics)
    
    print(f"\n✅ Rule-Based Results for AAPL:")
    print(f"   Discount Rate: {result_rule['discount_rate']:.2%}")
    print(f"   Terminal Growth: {result_rule['terminal_growth']:.2%}")
    print(f"   Projection Years: {result_rule['projection_years']}")
    print(f"\n{result_rule['explanation']}")
    
    # Test AI-powered (if API key available)
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key:
        print("\n[2/2] Testing AI-Powered Optimization...")
        optimizer_ai = AIParameterOptimizer(api_key=api_key, use_ai=True)
        result_ai = optimizer_ai.optimize_parameters('AAPL', sample_company, sample_metrics)
        
        print(f"\n✅ AI-Powered Results for AAPL:")
        print(f"   Discount Rate: {result_ai['discount_rate']:.2%}")
        print(f"   Terminal Growth: {result_ai['terminal_growth']:.2%}")
        print(f"   Projection Years: {result_ai['projection_years']}")
        print(f"\n{result_ai['explanation']}")
        
        if 'ai_reasoning' in result_ai:
            print(f"\n📝 AI Reasoning:\n{result_ai['ai_reasoning']}")
    else:
        print("\n[2/2] Skipping AI test (no GROQ_API_KEY found)")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)