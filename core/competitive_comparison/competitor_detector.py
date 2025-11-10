"""
competitor_detector.py

AI & fallback competitor detection. Supports Groq API (online) and Ollama (local).
"""

import json
import logging
from typing import List, Optional
import yfinance as yf

logger = logging.getLogger(__name__)


class CompetitorDetector:
    """
    Uses AI to identify the most relevant competitors for a stock.
    Falls back to curated/manual mappings when AI isn't available or fails.
    """

    def __init__(self, api_key: Optional[str] = None, use_ollama: bool = False):
        self.api_key = api_key
        self.use_ollama = use_ollama
        self.client = None

        if use_ollama:
            try:
                import ollama  # type: ignore
                ollama.list()  # sanity check that server is available
                self.client = "ollama"
                logger.info("✅ Competitor Detector initialized with Ollama (Local)")
            except ImportError:
                logger.error("❌ Ollama not installed. Run: pip install ollama")
                self.client = None
            except Exception as e:
                logger.error("❌ Ollama not running. Start it with: ollama serve")
                logger.debug(e)
                self.client = None
        elif api_key:
            try:
                from groq import Groq  # type: ignore
                self.client = Groq(api_key=api_key)
                logger.info("✅ Competitor Detector initialized with Groq API")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq: {e}")
                self.client = None

        if not self.client:
            logger.warning("⚠️ No AI available - will use manual competitor mappings")

    # --------------------
    # Public API
    # --------------------
    def find_competitors(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        industry: str,
        num_competitors: int = 2,
    ) -> List[str]:
        """
        Return a list of competitor tickers. Prefer AI + validation, otherwise fallback.
        """
        if not self.client:
            logger.warning("AI not available, using fallback competitor detection")
            return self._fallback_competitors(ticker, company_name, sector, industry)

        try:
            prompt = self._build_prompt(company_name, ticker, sector, industry, num_competitors)
            if self.use_ollama:
                ai_response = self._call_ollama(prompt)
            else:
                ai_response = self._call_groq(prompt)

            if not ai_response:
                logger.warning("AI returned no response, using fallback")
                return self._fallback_competitors(ticker, company_name, sector, industry)

            # Extract JSON
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                ai_response = ai_response.split('```')[1].split('```')[0].strip()

            result = json.loads(ai_response)
            competitors = result.get('competitors', [])
            reasoning = result.get('reasoning', '')

            # Validate tickers exist and are reasonable
            valid_competitors = []
            for comp in competitors[:num_competitors]:
                comp_ticker = comp.upper().strip()
                try:
                    t = yf.Ticker(comp_ticker)
                    info = t.info
                    if info and info.get('symbol'):
                        comp_name = info.get('longName', comp_ticker)
                        comp_sector = info.get('sector', 'Unknown')
                        if self._is_reasonable_competitor(ticker, company_name, comp_ticker, comp_name, sector, comp_sector):
                            valid_competitors.append(comp_ticker)
                            logger.info(f"✓ Validated competitor: {comp_ticker} ({comp_name})")
                        else:
                            logger.warning(f"⚠️ Questionable competitor: {comp_ticker}. Skipping.")
                except Exception as e:
                    logger.warning(f"Invalid ticker from AI: {comp_ticker} - {e}")

            if valid_competitors:
                logger.info(f"AI found competitors for {ticker}: {valid_competitors}")
                logger.debug(f"Reasoning: {reasoning}")
                return valid_competitors

            logger.warning("AI returned no valid competitors, using fallback")
            return self._fallback_competitors(ticker, company_name, sector, industry)

        except Exception as e:
            logger.error(f"AI competitor detection failed: {e}")
            return self._fallback_competitors(ticker, company_name, sector, industry)

    # --------------------
    # Helpers
    # --------------------
    def _build_prompt(self, company_name: str, ticker: str, sector: str, industry: str, num_competitors: int) -> str:
        return f"""You are a financial analyst identifying key competitors.

Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Identify the TOP {num_competitors} most relevant publicly-traded competitors.

CRITICAL RULES:
1. Focus on PRIMARY BUSINESS - ignore secondary operations
2. Companies must compete for the SAME CUSTOMERS in the SAME MARKET
3. Must be US-listed stocks with valid ticker symbols
4. Prefer similar market cap when possible
5. IGNORE sector classification if it doesn't match actual business

Respond with ONLY valid JSON in this exact format:
{{
  "competitors": ["TICKER1", "TICKER2"],
  "reasoning": "Brief explanation focusing on why these compete for the SAME customers/market"
}}

No other text, just the JSON.
"""

    def _call_groq(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create( # type: ignore
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": ("You are a financial analyst expert at identifying TRUE business competitors "
                                    "based on actual business overlap, not just sector labels. Respond ONLY with valid JSON.")
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()  # type: ignore
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        try:
            import ollama  # type: ignore
            response = ollama.chat(
                model='llama3.1',
                messages=[
                    {'role': 'system', 'content': 'You are a financial analyst expert. Respond ONLY with valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.2, 'num_predict': 300}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None

    def _is_reasonable_competitor(
        self,
        ticker: str,
        company_name: str,
        comp_ticker: str,
        comp_name: str,
        sector: str,
        comp_sector: str,
    ) -> bool:
        # Avoid self
        if comp_ticker == ticker:
            return False

        bad_pairs = {
            'TSLA': ['AMZN', 'WMT', 'TGT', 'COST'],
            'AAPL': ['WMT', 'TGT', 'COST'],
            'NFLX': ['AMZN', 'WMT'],
            'NKE': ['AMZN', 'WMT', 'TGT'],
        }
        if ticker in bad_pairs and comp_ticker in bad_pairs[ticker]:
            logger.warning(f"Blocked known bad pair: {ticker} vs {comp_ticker}")
            return False

        # Minimal further checks can be added later (market cap proximity, primary industry overlap)
        return True

    def _fallback_competitors(self, ticker: str, company_name: str, sector: str, industry: str) -> List[str]:
        known_competitors = {
            'TSLA': ['GM', 'F'],
            'GM': ['F', 'TSLA'],
            'F': ['GM', 'TSLA'],
            'RIVN': ['TSLA', 'F'],
            'LCID': ['TSLA', 'RIVN'],
            'AAPL': ['MSFT', 'GOOGL'],
            'MSFT': ['AAPL', 'GOOGL'],
            'GOOGL': ['MSFT', 'META'],
            'META': ['GOOGL', 'SNAP'],
            'AMZN': ['WMT', 'MSFT'],
            'NFLX': ['DIS', 'PARA'],
            'NVDA': ['AMD', 'INTC'],
            'AMD': ['NVDA', 'INTC'],
            'INTC': ['AMD', 'NVDA'],
            'WMT': ['TGT', 'COST'],
            'TGT': ['WMT', 'COST'],
            'COST': ['WMT', 'TGT'],
            'JPM': ['BAC', 'WFC'],
            'BAC': ['JPM', 'C'],
            'WFC': ['JPM', 'BAC'],
            'JNJ': ['PFE', 'ABBV'],
            'PFE': ['JNJ', 'MRK'],
            'BA': ['LMT', 'RTX'],
            'LMT': ['BA', 'RTX'],
            'XOM': ['CVX', 'COP'],
            'CVX': ['XOM', 'COP'],
            'NKE': ['ADDYY', 'UAA'],
            'V': ['MA', 'AXP'],
            'MA': ['V', 'AXP'],
        }

        if ticker in known_competitors:
            comps = known_competitors[ticker]
            logger.info(f"Using manual mapping for {ticker}: {comps}")
            return comps[:2]

        sector_competitors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Consumer Cyclical': ['HD', 'NKE', 'TGT', 'LOW'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX'],
            'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
            'Industrials': ['BA', 'CAT', 'GE', 'HON'],
            'Basic Materials': ['LIN', 'APD', 'ECL', 'DD'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D']
        }

        candidates = sector_competitors.get(sector, ['SPY', 'QQQ'])
        competitors = [t for t in candidates if t != ticker][:2]
        logger.info(f"Fallback competitors for {ticker} (sector: {sector}): {competitors}")
        return competitors
