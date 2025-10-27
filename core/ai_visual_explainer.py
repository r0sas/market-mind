# ai_visual_explainer.py
"""
AI-Powered Visual Explanations for Beginners
Generates simple, educational insights for tables and charts
"""

import logging
from typing import Dict, List, Optional
from groq import Groq
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIVisualExplainer:
    """
    Generates beginner-friendly explanations for financial visualizations.
    
    Explains:
    - Intrinsic value tables
    - Model comparison charts
    - Margin of safety analysis
    - Model selection charts
    - Parameter optimization results
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize explainer.
        
        Args:
            api_key: Groq API key for AI explanations
        """
        self.api_key = api_key
        
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                logger.info("AI Visual Explainer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {e}")
                self.client = None
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if AI explanations are available."""
        return self.client is not None
    
    def explain_intrinsic_value_table(
        self,
        ticker: str,
        current_price: float,
        valuations: Dict[str, float],
        average_value: float
    ) -> Optional[str]:
        """
        Explain the intrinsic value table for beginners.
        
        Args:
            ticker: Stock ticker
            current_price: Current market price
            valuations: Dictionary of model values
            average_value: Average intrinsic value
            
        Returns:
            Beginner-friendly explanation
        """
        if not self.is_available():
            return self._generate_simple_table_explanation(
                ticker, current_price, valuations, average_value
            )
        
        try:
            prompt = f"""You are explaining stock valuation to a complete beginner who has never invested before.

STOCK: {ticker}
Current Market Price: ${current_price:,.2f}

Calculated Intrinsic Values (what the stock is "really worth"):
{self._format_valuations(valuations)}

Average Intrinsic Value: ${average_value:,.2f}

Write a simple 3-4 sentence explanation that:
1. Explains what "intrinsic value" means in simple terms
2. Compares the current price to calculated values (is it expensive or cheap?)
3. Points out if there's agreement or disagreement between models
4. Gives a simple takeaway for a beginner

Use everyday language. Avoid jargon. Be encouraging and educational."""

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a patient financial educator explaining concepts to absolute beginners. Use simple, clear language and everyday analogies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate AI explanation: {e}")
            return self._generate_simple_table_explanation(
                ticker, current_price, valuations, average_value
            )
    
    def explain_comparison_chart(
        self,
        tickers: List[str],
        valuations: Dict[str, Dict[str, float]],
        current_prices: Dict[str, float]
    ) -> Optional[str]:
        """
        Explain the comparison chart showing multiple stocks.
        
        Args:
            tickers: List of stock tickers
            valuations: Dict of {ticker: {model: value}}
            current_prices: Dict of {ticker: price}
            
        Returns:
            Beginner-friendly explanation
        """
        if not self.is_available():
            return self._generate_simple_comparison_explanation(
                tickers, valuations, current_prices
            )
        
        try:
            # Calculate which stocks are undervalued/overvalued
            analysis = []
            for ticker in tickers:
                avg_value = sum(valuations[ticker].values()) / len(valuations[ticker])
                current = current_prices[ticker]
                diff_pct = ((avg_value - current) / current) * 100
                
                analysis.append(f"{ticker}: ${current:.2f} vs ${avg_value:.2f} ({diff_pct:+.1f}%)")
            
            prompt = f"""You are explaining a stock comparison chart to someone new to investing.

COMPARING {len(tickers)} STOCKS:
{chr(10).join(analysis)}

This chart shows multiple valuation models for each stock (the colored bars) and compares them to the current market price (the red line).

Write a 3-4 sentence explanation that:
1. Explains what the chart is showing in simple terms
2. Points out which stock(s) look like the best deals (most undervalued)
3. Mentions if any stocks look expensive (overvalued)
4. Reminds them this is just one factor in investing

Keep it simple and encouraging. Avoid technical jargon."""

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly investment teacher explaining concepts to beginners."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate comparison explanation: {e}")
            return self._generate_simple_comparison_explanation(
                tickers, valuations, current_prices
            )
    
    def explain_margin_of_safety(
        self,
        ticker: str,
        margin_pct: float,
        target_margin: float,
        is_undervalued: bool
    ) -> Optional[str]:
        """
        Explain margin of safety concept for beginners.
        
        Args:
            ticker: Stock ticker
            margin_pct: Current margin of safety percentage
            target_margin: Target margin (e.g., 25%)
            is_undervalued: Whether stock meets margin requirement
            
        Returns:
            Beginner-friendly explanation
        """
        if not self.is_available():
            return self._generate_simple_margin_explanation(
                ticker, margin_pct, target_margin, is_undervalued
            )
        
        try:
            prompt = f"""Explain "margin of safety" to someone who has never invested before.

STOCK: {ticker}
Current Margin of Safety: {margin_pct:.1f}%
Target Margin: {target_margin:.1f}%
Status: {"✓ Good deal" if is_undervalued else "✗ Not a good deal right now"}

The margin of safety is like a "safety cushion" - how much cheaper the stock is than what we think it's worth.

Write a 2-3 sentence explanation that:
1. Uses a simple analogy (like buying a $100 item on sale)
2. Explains if this is a good or bad margin for {ticker}
3. Tells them what they should look for

Use very simple language."""

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are explaining investing basics using simple analogies and everyday examples."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=250
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate margin explanation: {e}")
            return self._generate_simple_margin_explanation(
                ticker, margin_pct, target_margin, is_undervalued
            )
    
    def explain_model_selection(
        self,
        ticker: str,
        selected_models: List[str],
        excluded_models: List[str],
        reasons: Dict[str, str]
    ) -> Optional[str]:
        """
        Explain why certain models were selected or excluded.
        
        Args:
            ticker: Stock ticker
            selected_models: Models that were selected
            excluded_models: Models that were excluded
            reasons: Reasons for exclusions
            
        Returns:
            Beginner-friendly explanation
        """
        if not self.is_available():
            return self._generate_simple_selection_explanation(
                ticker, selected_models, excluded_models
            )
        
        try:
            prompt = f"""Explain to a beginner why we use different valuation models for different companies.

COMPANY: {ticker}
Using these models: {', '.join(selected_models)} ({len(selected_models)} total)
Not using: {', '.join(excluded_models)} ({len(excluded_models)} total)

Main reason models were excluded: {list(reasons.values())[0] if reasons else 'Data availability'}

Write a 2-3 sentence explanation that:
1. Explains why not all models work for all companies (use simple analogy)
2. Mentions what makes {ticker} suitable for the selected models
3. Reassures them this is normal and actually better for accuracy

Keep it simple and positive."""

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a patient teacher explaining financial concepts to complete beginners."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=250
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate selection explanation: {e}")
            return self._generate_simple_selection_explanation(
                ticker, selected_models, excluded_models
            )
    
    def explain_optimized_parameters(
        self,
        ticker: str,
        discount_rate: float,
        terminal_growth: float,
        explanation: str
    ) -> Optional[str]:
        """
        Explain optimized parameters for beginners.
        
        Args:
            ticker: Stock ticker
            discount_rate: Optimized discount rate
            terminal_growth: Optimized terminal growth
            explanation: Technical explanation from optimizer
            
        Returns:
            Beginner-friendly explanation
        """
        if not self.is_available():
            return self._generate_simple_parameter_explanation(
                ticker, discount_rate, terminal_growth
            )
        
        try:
            prompt = f"""Explain what "discount rate" and "terminal growth" mean to someone new to investing.

COMPANY: {ticker}
Discount Rate: {discount_rate:.1%}
Terminal Growth: {terminal_growth:.1%}

Technical explanation: {explanation}

Write a 2-3 sentence explanation that:
1. Explains these are like "risk settings" - higher risk = higher discount rate
2. Mentions if {ticker} is considered low-risk, medium-risk, or high-risk based on the {discount_rate:.1%} rate
3. Uses a simple analogy

Avoid technical jargon. Be encouraging."""

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are explaining financial concepts using simple, everyday language and analogies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=250
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate parameter explanation: {e}")
            return self._generate_simple_parameter_explanation(
                ticker, discount_rate, terminal_growth
            )
    
    # Fallback methods (when AI is unavailable)
    
    def _generate_simple_table_explanation(
        self, ticker: str, current_price: float, valuations: Dict, average: float
    ) -> str:
        """Generate simple explanation without AI."""
        diff = average - current_price
        diff_pct = (diff / current_price) * 100
        
        if diff_pct > 20:
            verdict = f"appears undervalued - trading at ${current_price:.2f} but worth around ${average:.2f}"
        elif diff_pct < -20:
            verdict = f"appears overvalued - trading at ${current_price:.2f} but worth around ${average:.2f}"
        else:
            verdict = f"is fairly valued - trading at ${current_price:.2f}, close to its estimated worth of ${average:.2f}"
        
        range_vals = list(valuations.values())
        spread = max(range_vals) - min(range_vals)
        
        if spread / average > 0.3:
            agreement = "There's significant disagreement between models, so take these estimates with caution."
        else:
            agreement = "The models mostly agree on the value, which increases confidence."
        
        return f"""**What This Means:** {ticker} {verdict}. We used {len(valuations)} different valuation methods to estimate what the stock is "really worth" (its intrinsic value). {agreement} Remember, this is just one factor to consider when investing!"""
    
    def _generate_simple_comparison_explanation(
        self, tickers: List[str], valuations: Dict, current_prices: Dict
    ) -> str:
        """Generate simple comparison without AI."""
        undervalued = []
        overvalued = []
        
        for ticker in tickers:
            avg = sum(valuations[ticker].values()) / len(valuations[ticker])
            current = current_prices[ticker]
            
            if avg > current * 1.15:
                undervalued.append(ticker)
            elif current > avg * 1.15:
                overvalued.append(ticker)
        
        text = f"**Comparing {len(tickers)} Stocks:** This chart shows what each stock is currently trading at (red line) versus what our analysis suggests it's worth (colored bars). "
        
        if undervalued:
            text += f"{', '.join(undervalued)} look{'s' if len(undervalued) == 1 else ''} like potential bargains. "
        if overvalued:
            text += f"{', '.join(overvalued)} seem{'s' if len(overvalued) == 1 else ''} expensive right now. "
        
        text += "Remember: intrinsic value is just one piece of the puzzle!"
        
        return text
    
    def _generate_simple_margin_explanation(
        self, ticker: str, margin_pct: float, target: float, is_good: bool
    ) -> str:
        """Generate simple margin explanation without AI."""
        
        analogy = f"Think of it like buying a ${100:.0f} item on sale for ${100 - target*100:.0f} - that's a {target*100:.0f}% discount giving you a safety cushion."
        
        if is_good:
            verdict = f"{ticker} has a {margin_pct:.0f}% margin of safety, which meets the {target*100:.0f}% target. This is a good safety cushion!"
        else:
            verdict = f"{ticker}'s margin of safety is {margin_pct:.0f}%, below the {target*100:.0f}% target. You might want to wait for a better price."
        
        return f"""**Margin of Safety Explained:** {analogy} {verdict}"""
    
    def _generate_simple_selection_explanation(
        self, ticker: str, selected: List[str], excluded: List[str]
    ) -> str:
        """Generate simple model selection explanation without AI."""
        
        return f"""**Why Different Models?** Just like you wouldn't use a hammer for every job, we use different valuation methods for different companies. For {ticker}, {len(selected)} models are appropriate based on its financial characteristics, while {len(excluded)} don't fit well. Using the right tools gives more accurate results!"""
    
    def _generate_simple_parameter_explanation(
        self, ticker: str, discount_rate: float, terminal_growth: float
    ) -> str:
        """Generate simple parameter explanation without AI."""
        
        if discount_rate < 0.09:
            risk = "low-risk (like a steady utility company)"
        elif discount_rate < 0.12:
            risk = "moderate-risk (typical for established companies)"
        else:
            risk = "higher-risk (like a fast-growing tech company)"
        
        return f"""**Risk Settings:** The discount rate ({discount_rate:.1%}) is like a "risk dial" - {ticker} is considered {risk}. Lower risk companies need lower returns to be attractive investments. These customized settings help us get more accurate valuations for each unique company."""
    
    def _format_valuations(self, valuations: Dict[str, float]) -> str:
        """Format valuations for prompt."""
        return "\n".join([f"- {model}: ${value:,.2f}" for model, value in valuations.items()])


# Convenience function
def get_visual_explanation(
    explanation_type: str,
    api_key: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Quick function to get visual explanations.
    
    Args:
        explanation_type: Type of explanation needed
        api_key: Groq API key
        **kwargs: Data for explanation
        
    Returns:
        Explanation text or None
    """
    explainer = AIVisualExplainer(api_key=api_key)
    
    if explanation_type == "table":
        return explainer.explain_intrinsic_value_table(**kwargs)
    elif explanation_type == "comparison":
        return explainer.explain_comparison_chart(**kwargs)
    elif explanation_type == "margin":
        return explainer.explain_margin_of_safety(**kwargs)
    elif explanation_type == "selection":
        return explainer.explain_model_selection(**kwargs)
    elif explanation_type == "parameters":
        return explainer.explain_optimized_parameters(**kwargs)
    else:
        return None


# Example usage
if __name__ == "__main__":
    import os
    
    print("="*70)
    print("AI VISUAL EXPLAINER TEST")
    print("="*70)
    
    api_key = os.getenv("GROQ_API_KEY")
    explainer = AIVisualExplainer(api_key=api_key)
    
    # Test 1: Intrinsic Value Table
    print("\n[1/5] Testing Intrinsic Value Table Explanation...")
    explanation = explainer.explain_intrinsic_value_table(
        ticker="AAPL",
        current_price=180.50,
        valuations={
            "DCF": 195.00,
            "DDM": 185.00,
            "P/E Model": 200.00
        },
        average_value=193.33
    )
    print(f"\n{explanation}\n")
    
    # Test 2: Margin of Safety
    print("\n[2/5] Testing Margin of Safety Explanation...")
    explanation = explainer.explain_margin_of_safety(
        ticker="AAPL",
        margin_pct=7.1,
        target_margin=25.0,
        is_undervalued=False
    )
    print(f"\n{explanation}\n")
    
    # Test 3: Model Selection
    print("\n[3/5] Testing Model Selection Explanation...")
    explanation = explainer.explain_model_selection(
        ticker="TSLA",
        selected_models=["DCF", "P/E Model"],
        excluded_models=["DDM Single", "DDM Multi"],
        reasons={"DDM Single": "Company does not pay dividends"}
    )
    print(f"\n{explanation}\n")
    
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)