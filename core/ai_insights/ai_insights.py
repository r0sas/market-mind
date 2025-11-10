# ai_insights.py
"""
AI-powered insights for stock valuations using Groq or Ollama API.

Generates structured contextual analysis of valuation results.
"""

import os
import logging
from typing import Dict, Optional, List, Union
from core.ai_provider.client import UnifiedAIClient, AIProvider
from core.ai_provider.enums import AIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIInsightsGenerator:
    """
    Generate AI-powered insights for stock valuations.
    
    Supports Groq API (Llama 3.1) or local Ollama fallback.
    Provides structured JSON output for integration with dashboards.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        use_ollama: bool = False, 
        model: Optional[str] = None
    ):
        """Initialize AI insights generator."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.use_ollama = use_ollama

        if use_ollama:
            provider = AIProvider.OLLAMA
            default_model = "llama3.1"
        elif self.api_key:
            provider = AIProvider.GROQ
            default_model = "llama-3.1-8b-instant"
        else:
            provider = AIProvider.NONE
            default_model = None

        self.model = model or default_model
        self.ai_client = UnifiedAIClient(provider, self.api_key, self.model)
        self.client = self.ai_client if self.ai_client.is_available else None

        if self.ai_client.is_available:
            logger.info(f"AI initialized: {self.ai_client.get_provider_name()} with {self.model}")
        else:
            logger.warning("No AI provider available")

    def is_available(self) -> bool:
        """Check if AI insights are available."""
        return self.ai_client.is_available

    def generate_insights(
        self,
        ticker: str,
        current_price: float,
        valuations: Dict[str, float],
        sector: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        confidence_scores: Optional[Dict[str, str]] = None
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Generate AI-powered insights for a single stock.
        
        Returns structured JSON with:
        - ai_text: raw AI response
        - summary: structured over/under/fair valuation and key considerations
        - metrics: computed averages and ranges
        """
        if not self.is_available():
            logger.warning("AI not available, returning fallback summary")
            return self._fallback_insights(ticker, current_price, valuations)

        try:
            prompt = self._build_prompt(ticker, current_price, valuations, sector, warnings, confidence_scores)
            ai_text = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Provide concise, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )

            if not ai_text:
                raise ValueError("AI returned empty response")

            metrics = self._compute_metrics(current_price, valuations)
            summary = self._parse_ai_summary(ai_text, metrics)
            
            return {
                "ticker": ticker,
                "ai_text": ai_text,
                "summary": summary,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Failed to generate AI insights for {ticker}: {e}")
            return self._fallback_insights(ticker, current_price, valuations)

    def generate_batch_insights(self, stocks_data: List[Dict]) -> Dict[str, Dict]:
        """
        Generate insights for multiple stocks.
        """
        results = {}
        for stock in stocks_data:
            ticker = stock.get("ticker")
            if not ticker:
                continue
            results[ticker] = self.generate_insights(
                ticker=ticker,
                current_price=stock.get("current_price"),
                valuations=stock.get("valuations", {}),
                sector=stock.get("sector"),
                warnings=stock.get("warnings"),
                confidence_scores=stock.get("confidence_scores")
            )
        return results

    def test_connection(self) -> bool:
        """Test if AI provider is working."""
        return self.ai_client.test_connection()

    # ----------------------------
    # Internal helper methods
    # ----------------------------
    
    def _build_prompt(self, ticker, current_price, valuations, sector, warnings, confidence_scores) -> str:
        """Construct AI prompt for a stock."""
        values = list(valuations.values())
        avg_value = sum(values) / len(values) if values else 0
        min_value, max_value = (min(values), max(values)) if values else (0, 0)
        overvaluation_pct = ((current_price - avg_value) / avg_value * 100) if avg_value else 0

        valuation_text = "\n".join([f"  - {k}: ${v:,.2f}" for k, v in valuations.items()])

        confidence_section = ""
        if confidence_scores:
            confidence_section = "\n\nModel Confidence Levels:\n" + "\n".join([f"  - {k}: {v} confidence" for k, v in confidence_scores.items()])

        warnings_section = ""
        if warnings:
            warnings_section = "\n\nData Quality Warnings:\n" + "\n".join([f"  - {w}" for w in warnings[:3]])

        sector_text = f"\nSector: {sector}" if sector else ""

        prompt = f"""Analyze stock {ticker}:{sector_text}
Current Market Price: ${current_price:,.2f}
Intrinsic Value Estimates:
{valuation_text}
Average Intrinsic Value: ${avg_value:,.2f}
Valuation Range: ${min_value:,.2f} - ${max_value:,.2f}
Market Premium/Discount: {overvaluation_pct:+.1f}%{confidence_section}{warnings_section}
Provide a 3-4 sentence analysis. Focus on over/under/fair valuation, key drivers, and actionable advice."""
        return prompt

    def _compute_metrics(self, current_price, valuations) -> Dict[str, float]:
        """Compute average, min/max, and over/under valuation."""
        values = list(valuations.values())
        avg_value = sum(values) / len(values) if values else 0
        min_value, max_value = (min(values), max(values)) if values else (0, 0)
        overvaluation_pct = ((current_price - avg_value) / avg_value * 100) if avg_value else 0
        return {
            "average_value": avg_value,
            "min_value": min_value,
            "max_value": max_value,
            "overvaluation_pct": overvaluation_pct
        }

    def _parse_ai_summary(self, ai_text: str, metrics: Dict[str, float]) -> Dict[str, str]:
        """Extract key structured points from AI text (simple placeholder)."""
        # Could integrate NLP parsing here; for now, just include first sentence
        lines = ai_text.split(".")
        key_point = lines[0].strip() if lines else ""
        return {
            "valuation_summary": key_point,
            "recommendation": ai_text
        }

    def _fallback_insights(self, ticker: str, current_price: float, valuations: Dict[str, float]) -> Dict:
        """Fallback structured insights if AI is unavailable."""
        metrics = self._compute_metrics(current_price, valuations)
        status = "Fairly valued"
        if metrics["overvaluation_pct"] > 5:
            status = "Overvalued"
        elif metrics["overvaluation_pct"] < -5:
            status = "Undervalued"
        return {
            "ticker": ticker,
            "ai_text": "AI unavailable, fallback summary used.",
            "summary": {"valuation_summary": status, "recommendation": "Use caution and consider metrics."},
            "metrics": metrics
        }


# Convenience function
def get_ai_insights(ticker: str, current_price: float, valuations: Dict[str, float], api_key: Optional[str] = None, **kwargs) -> Dict:
    generator = AIInsightsGenerator(api_key=api_key)
    return generator.generate_insights(ticker, current_price, valuations, **kwargs)
