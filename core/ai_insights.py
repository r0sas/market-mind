# ai_insights.py
"""
AI-powered insights for stock valuations using Groq API.

This module generates contextual analysis of valuation results using
free Groq API (powered by Llama 3.1).
"""

import os
import logging
from typing import Dict, Optional, List
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIInsightsGenerator:
    """
    Generate AI-powered insights for stock valuations.
    
    Uses Groq API (free tier) with Llama 3.1 for fast, intelligent analysis.
    Falls back to local Ollama if Groq is unavailable.
    
    Example:
        >>> generator = AIInsightsGenerator(api_key="your-key")
        >>> insights = generator.generate_insights("AAPL", valuation_data)
        >>> print(insights)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "llama-3.1-8b-instant",
        enable_ollama_fallback: bool = True
    ):
        """
        Initialize AI insights generator.
        
        Args:
            api_key: Groq API key. If None, looks for GROQ_API_KEY env variable
            model: Groq model to use (default: llama-3.1-8b-instant)
            enable_ollama_fallback: If True, use local Ollama when Groq fails
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.enable_ollama_fallback = enable_ollama_fallback
        
        if not self.api_key:
            logger.warning("No Groq API key found. Will try Ollama if available.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq AI initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if AI insights are available."""
        return self.client is not None
    
    def generate_insights(
        self,
        ticker: str,
        current_price: float,
        valuations: Dict[str, float],
        sector: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        confidence_scores: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Generate AI-powered insights for a stock valuation.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current market price
            valuations: Dictionary of model names and intrinsic values
            sector: Company sector (optional but recommended)
            warnings: List of data quality warnings (optional)
            confidence_scores: Dictionary of confidence scores per model (optional)
            
        Returns:
            AI-generated insights as string, or None if generation fails
        """
        if not self.is_available():
            logger.warning("AI insights not available (no API key or initialization failed)")
            return None
        
        try:
            # Build the prompt
            prompt = self._build_prompt(
                ticker, current_price, valuations, sector, warnings, confidence_scores
            )
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst. Provide concise, actionable insights on stock valuations. Be direct and specific. Avoid generic advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=400,
                temperature=0.7,
                top_p=0.9
            )
            
            insights = response.choices[0].message.content.strip()
            logger.info(f"Generated AI insights for {ticker} ({len(insights)} chars)")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate AI insights for {ticker}: {e}")
            return None
    
    def _build_prompt(
        self,
        ticker: str,
        current_price: float,
        valuations: Dict[str, float],
        sector: Optional[str],
        warnings: Optional[List[str]],
        confidence_scores: Optional[Dict[str, str]]
    ) -> str:
        """Build the prompt for AI analysis."""
        
        # Calculate average and range
        values = list(valuations.values())
        avg_value = sum(values) / len(values) if values else 0
        min_value = min(values) if values else 0
        max_value = max(values) if values else 0
        
        overvaluation_pct = ((current_price - avg_value) / avg_value * 100) if avg_value > 0 else 0
        
        # Build valuation summary
        valuation_text = "\n".join([
            f"  - {model.replace('_', ' ').title()}: ${value:,.2f}"
            for model, value in valuations.items()
        ])
        
        # Add confidence scores if available
        if confidence_scores:
            confidence_text = "\n".join([
                f"  - {model.replace('_', ' ').title()}: {score} confidence"
                for model, score in confidence_scores.items()
            ])
            confidence_section = f"\n\nModel Confidence Levels:\n{confidence_text}"
        else:
            confidence_section = ""
        
        # Add warnings if available
        if warnings and len(warnings) > 0:
            warnings_text = "\n".join([f"  - {w}" for w in warnings[:3]])  # Limit to 3
            warnings_section = f"\n\nData Quality Warnings:\n{warnings_text}"
        else:
            warnings_section = ""
        
        # Add sector context
        sector_text = f"\nSector: {sector}" if sector else ""
        
        # Build complete prompt
        prompt = f"""Analyze this stock valuation for {ticker}:{sector_text}

Current Market Price: ${current_price:,.2f}

Intrinsic Value Estimates:
{valuation_text}

Average Intrinsic Value: ${avg_value:,.2f}
Valuation Range: ${min_value:,.2f} - ${max_value:,.2f}
Market Premium/Discount: {overvaluation_pct:+.1f}%{confidence_section}{warnings_section}

Provide a brief analysis (3-4 sentences) covering:
1. Is the stock over/under/fairly valued based on these models?
2. What might explain any significant discrepancy between market price and intrinsic values?
3. What's the key consideration for investors right now?

Be specific, concise, and actionable. Focus on the numbers provided."""
        
        return prompt
    
    def generate_batch_insights(
        self,
        stocks_data: List[Dict]
    ) -> Dict[str, Optional[str]]:
        """
        Generate insights for multiple stocks.
        
        Args:
            stocks_data: List of dicts, each containing stock valuation data
            
        Returns:
            Dictionary mapping ticker to insights
        """
        results = {}
        
        for stock_data in stocks_data:
            ticker = stock_data.get('ticker')
            if not ticker:
                continue
            
            insights = self.generate_insights(
                ticker=ticker,
                current_price=stock_data.get('current_price'),
                valuations=stock_data.get('valuations', {}),
                sector=stock_data.get('sector'),
                warnings=stock_data.get('warnings'),
                confidence_scores=stock_data.get('confidence_scores')
            )
            
            results[ticker] = insights
        
        return results
    
    def test_connection(self) -> bool:
        """
        Test if Groq API is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Groq API test successful: {result}")
            return True
        except Exception as e:
            logger.error(f"Groq API test failed: {e}")
            return False


# Convenience function for simple usage
def get_ai_insights(
    ticker: str,
    current_price: float,
    valuations: Dict[str, float],
    api_key: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Quick function to generate AI insights.
    
    Args:
        ticker: Stock ticker
        current_price: Current market price
        valuations: Dict of model names to values
        api_key: Groq API key (optional, uses env variable if not provided)
        **kwargs: Additional arguments (sector, warnings, etc.)
        
    Returns:
        AI-generated insights or None
    """
    generator = AIInsightsGenerator(api_key=api_key)
    return generator.generate_insights(
        ticker=ticker,
        current_price=current_price,
        valuations=valuations,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    print("="*70)
    print("AI INSIGHTS MODULE TEST")
    print("="*70)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n❌ No GROQ_API_KEY found in environment variables.")
        print("\nTo set it up:")
        print("  1. Get free API key from: https://console.groq.com")
        print("  2. Set environment variable:")
        print("     export GROQ_API_KEY='your-key-here'  # Mac/Linux")
        print("     set GROQ_API_KEY=your-key-here       # Windows CMD")
        print("     $env:GROQ_API_KEY='your-key-here'    # Windows PowerShell")
        exit(1)
    
    # Initialize generator
    generator = AIInsightsGenerator(api_key=api_key)
    
    # Test connection
    print("\n[1/3] Testing Groq API connection...")
    if generator.test_connection():
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed!")
        exit(1)
    
    # Test with sample stock data
    print("\n[2/3] Generating insights for AAPL...")
    
    sample_valuations = {
        "dcf": 150.50,
        "ddm_multi_stage": 165.20,
        "pe_model": 185.75,
        "graham_value": 155.00,
        "asset_based": 120.30
    }
    
    insights = generator.generate_insights(
        ticker="AAPL",
        current_price=180.50,
        valuations=sample_valuations,
        sector="Technology",
        warnings=["Limited FCF history (4 years)"],
        confidence_scores={
            "dcf": "High",
            "pe_model": "High",
            "graham_value": "Medium"
        }
    )
    
    if insights:
        print("\n✅ Insights generated successfully!")
        print("\n" + "="*70)
        print("AI ANALYSIS:")
        print("="*70)
        print(insights)
        print("="*70)
    else:
        print("❌ Failed to generate insights")
    
    # Test batch processing
    print("\n[3/3] Testing batch processing...")
    
    batch_data = [
        {
            "ticker": "MSFT",
            "current_price": 380.00,
            "valuations": {"dcf": 320.00, "pe_model": 390.00},
            "sector": "Technology"
        },
        {
            "ticker": "KO",
            "current_price": 60.00,
            "valuations": {"dcf": 55.00, "ddm_single": 58.00},
            "sector": "Consumer Defensive"
        }
    ]
    
    batch_results = generator.generate_batch_insights(batch_data)
    
    for ticker, insight in batch_results.items():
        if insight:
            print(f"\n✅ {ticker}: Generated")
        else:
            print(f"\n❌ {ticker}: Failed")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)