import logging
from typing import Dict, List, Optional
from core.ai_provider.client import UnifiedAIClient
from core.ai_provider.enums import AIProvider
from .simple_fallbacks import (
    generate_simple_table_explanation,
    generate_simple_comparison_explanation,
    generate_simple_margin_explanation,
    generate_simple_selection_explanation,
    generate_simple_parameter_explanation
)
from .utils import format_valuations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AIVisualExplainer:
    """
    Generates beginner-friendly explanations for financial visualizations.
    """

    def __init__(self, api_key: Optional[str] = None, use_ollama: bool = False):
        self.api_key = api_key
        self.use_ollama = use_ollama
        provider = AIProvider.OLLAMA if use_ollama else AIProvider.GROQ if api_key else AIProvider.NONE
        self.ai_client = UnifiedAIClient(
            provider,
            api_key,
            model="llama3.1" if use_ollama else "llama-3.1-8b-instant"
        )
        self.client = self.ai_client if self.ai_client.is_available else None
        if self.ai_client.is_available:
            logger.info(f"AI Visual Explainer initialized: {self.ai_client.get_provider_name()}")

    def is_available(self) -> bool:
        return self.ai_client.is_available

    # ------------------- Explanations ------------------- #
    def explain_intrinsic_value_table(
        self, ticker: str, current_price: float, valuations: Dict[str, float], average_value: float
    ) -> Optional[str]:
        if not self.is_available():
            return generate_simple_table_explanation(ticker, current_price, valuations, average_value)

        try:
            prompt = f"""You are explaining stock valuation to a complete beginner.
STOCK: {ticker}
Current Market Price: ${current_price:,.2f}
Calculated Intrinsic Values:
{format_valuations(valuations)}
Average Intrinsic Value: ${average_value:,.2f}

Write a simple 3-4 sentence explanation that is clear and encouraging."""
            
            explanation = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a patient financial educator for beginners."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            if not explanation:
                return generate_simple_table_explanation(ticker, current_price, valuations, average_value)
            return explanation

        except Exception as e:
            logger.error(f"Failed to generate AI explanation: {e}")
            return generate_simple_table_explanation(ticker, current_price, valuations, average_value)

    def explain_comparison_chart(self, tickers: List[str], valuations: Dict[str, Dict[str, float]], current_prices: Dict[str, float]) -> Optional[str]:
        if not self.is_available():
            return generate_simple_comparison_explanation(tickers, valuations, current_prices)
        
        try:
            analysis = [
                f"{t}: ${current_prices[t]:.2f} vs ${sum(valuations[t].values())/len(valuations[t]):.2f} "
                f"({((sum(valuations[t].values())/len(valuations[t])-current_prices[t])/current_prices[t])*100:+.1f}%)"
                for t in tickers
            ]
            prompt = f"""You are explaining a stock comparison chart to a beginner.
COMPARING {len(tickers)} STOCKS:
{chr(10).join(analysis)}

Write a 3-4 sentence explanation that is simple and encouraging."""
            
            explanation = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a friendly investment teacher for beginners."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            if not explanation:
                return generate_simple_comparison_explanation(tickers, valuations, current_prices)
            return explanation

        except Exception as e:
            logger.error(f"Failed to generate comparison explanation: {e}")
            return generate_simple_comparison_explanation(tickers, valuations, current_prices)

    def explain_margin_of_safety(self, ticker: str, margin_pct: float, target_margin: float, is_undervalued: bool) -> Optional[str]:
        if not self.is_available():
            return generate_simple_margin_explanation(ticker, margin_pct, target_margin, is_undervalued)
        try:
            prompt = f"""Explain 'margin of safety' to a beginner.
STOCK: {ticker}, Current Margin: {margin_pct:.1f}%, Target: {target_margin:.1f}%
Status: {'✓ Good deal' if is_undervalued else '✗ Not a good deal'}
Write a 2-3 sentence explanation using a simple analogy."""
            
            explanation = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "You explain investing basics using simple analogies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            if not explanation:
                return generate_simple_margin_explanation(ticker, margin_pct, target_margin, is_undervalued)
            return explanation

        except Exception as e:
            logger.error(f"Failed to generate margin explanation: {e}")
            return generate_simple_margin_explanation(ticker, margin_pct, target_margin, is_undervalued)

    def explain_model_selection(self, ticker: str, selected_models: List[str], excluded_models: List[str], reasons: Dict[str, str]) -> Optional[str]:
        if not self.is_available():
            return generate_simple_selection_explanation(ticker, selected_models, excluded_models)
        try:
            prompt = f"""Explain why certain models were selected for {ticker}.
Selected: {', '.join(selected_models)}, Excluded: {', '.join(excluded_models)}
Main reason for exclusion: {list(reasons.values())[0] if reasons else 'Data availability'}
Write 2-3 sentences in simple language."""
            
            explanation = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a patient teacher explaining finance to beginners."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            if not explanation:
                return generate_simple_selection_explanation(ticker, selected_models, excluded_models)
            return explanation

        except Exception as e:
            logger.error(f"Failed to generate selection explanation: {e}")
            return generate_simple_selection_explanation(ticker, selected_models, excluded_models)

    def explain_optimized_parameters(self, ticker: str, discount_rate: float, terminal_growth: float, explanation: str) -> Optional[str]:
        if not self.is_available():
            return generate_simple_parameter_explanation(ticker, discount_rate, terminal_growth)
        try:
            prompt = f"""Explain discount rate and terminal growth for {ticker}.
Discount Rate: {discount_rate:.1%}, Terminal Growth: {terminal_growth:.1%}
Technical Explanation: {explanation}
Write 2-3 sentences using simple analogies."""
            
            explanation = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "Explain finance using simple, everyday language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            if not explanation:
                return generate_simple_parameter_explanation(ticker, discount_rate, terminal_growth)
            return explanation

        except Exception as e:
            logger.error(f"Failed to generate parameter explanation: {e}")
            return generate_simple_parameter_explanation(ticker, discount_rate, terminal_growth)
