# competitive_comparison.py - COMPLETE FIXED VERSION WITH OLLAMA SUPPORT
"""
AI-Powered Competitive Comparison Dashboard
Uses AI (Groq or Ollama) to identify competitors and generate comparison analysis
"""

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class CompetitorDetector:
    """
    Uses AI to identify the most relevant competitors for a stock.
    Supports both Groq API (online) and Ollama (local Llama3).
    """
    
    def __init__(self, api_key: Optional[str] = None, use_ollama: bool = False):
        """
        Initialize competitor detector.
        
        Args:
            api_key: Groq API key for AI competitor detection
            use_ollama: If True, use Ollama (local) instead of Groq
        """
        self.api_key = api_key
        self.use_ollama = use_ollama
        self.client = None
        
        if use_ollama:
            # Try to use Ollama (local)
            try:
                import ollama
                # Test if Ollama is running
                ollama.list()
                self.client = "ollama"  # Marker that we're using Ollama
                logger.info("✅ Competitor Detector initialized with Ollama (Local)")
            except ImportError:
                logger.error("❌ Ollama not installed. Run: pip install ollama")
                self.client = None
            except Exception as e:
                logger.error(f"❌ Ollama not running. Start it with: ollama serve")
                logger.error(f"Error: {e}")
                self.client = None
        elif api_key:
            # Try to use Groq (online)
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                logger.info("✅ Competitor Detector initialized with Groq API")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq: {e}")
                self.client = None
        
        if not self.client:
            logger.warning("⚠️ No AI available - will use manual competitor mappings")
    
    def find_competitors(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        industry: str,
        num_competitors: int = 2
    ) -> List[str]:
        """
        Use AI to find the most relevant competitors.
        
        Args:
            ticker: Stock ticker
            company_name: Company name
            sector: Company sector
            industry: Company industry
            num_competitors: Number of competitors to return (1 or 2)
            
        Returns:
            List of competitor tickers
        """
        if not self.client:
            logger.warning("AI not available, using fallback competitor detection")
            return self._fallback_competitors(ticker, company_name, sector, industry)
        
        try:
            prompt = f"""You are a financial analyst identifying key competitors.

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

Examples of GOOD competitor pairs:
- Tesla (TSLA) → GM, F (both make cars, NOT Amazon/Walmart)
- Netflix (NFLX) → DIS, PARA (streaming, NOT retail)
- Nike (NKE) → ADDYY, UAA (athletic apparel, NOT general retail)

For {company_name}:
- What is their PRIMARY product/service?
- Who directly competes for those same customers?

Respond with ONLY valid JSON in this exact format:
{{
  "competitors": ["TICKER1", "TICKER2"],
  "reasoning": "Brief explanation focusing on why these compete for the SAME customers/market"
}}

No other text, just the JSON."""

            # Call appropriate AI service
            if self.use_ollama:
                ai_response = self._call_ollama(prompt)
            else:
                ai_response = self._call_groq(prompt)
            
            if not ai_response:
                logger.warning("AI returned no response, using fallback")
                return self._fallback_competitors(ticker, company_name, sector, industry)
            
            # Extract JSON from response
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                ai_response = ai_response.split('```')[1].split('```')[0].strip()
            
            result = json.loads(ai_response)
            competitors = result.get('competitors', [])
            reasoning = result.get('reasoning', '')
            
            # Validate tickers exist
            valid_competitors = []
            for comp_ticker in competitors[:num_competitors]:
                try:
                    comp_ticker = comp_ticker.upper().strip()
                    # Quick check if ticker exists
                    test = yf.Ticker(comp_ticker)
                    info = test.info
                    if info and 'symbol' in info:
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
                logger.info(f"Reasoning: {reasoning}")
                return valid_competitors
            else:
                logger.warning("AI returned no valid competitors, using fallback")
                return self._fallback_competitors(ticker, company_name, sector, industry)
            
        except Exception as e:
            logger.error(f"AI competitor detection failed: {e}")
            return self._fallback_competitors(ticker, company_name, sector, industry)
    
    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at identifying TRUE business competitors based on actual business overlap, not just sector labels. Respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama (local Llama3)"""
        try:
            import ollama
            
            response = ollama.chat(
                model='llama3.1',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial analyst expert at identifying TRUE business competitors. Respond ONLY with valid JSON.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.2,
                    'num_predict': 300
                }
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
        comp_sector: str
    ) -> bool:
        """
        Sanity check to filter out obviously wrong competitors.
        
        Returns:
            True if competitor seems reasonable, False otherwise
        """
        # Don't compare company to itself
        if comp_ticker == ticker:
            return False
        
        # Specific known bad matches (retail vs auto, etc.)
        bad_pairs = {
            'TSLA': ['AMZN', 'WMT', 'TGT', 'COST'],  # Tesla ≠ Retail
            'AAPL': ['WMT', 'TGT', 'COST'],          # Apple ≠ General Retail
            'NFLX': ['AMZN', 'WMT'],                 # Netflix ≠ General Retail
            'NKE': ['AMZN', 'WMT', 'TGT'],           # Nike ≠ General Retail
        }
        
        if ticker in bad_pairs and comp_ticker in bad_pairs[ticker]:
            logger.warning(f"Blocked known bad pair: {ticker} vs {comp_ticker}")
            return False
        
        return True
    
    def _fallback_competitors(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        industry: str
    ) -> List[str]:
        """
        Improved fallback competitor detection using manual mappings.
        """
        # SPECIFIC COMPANY MAPPINGS (manually curated for common stocks)
        known_competitors = {
            # Auto/EV
            'TSLA': ['GM', 'F'],
            'GM': ['F', 'TSLA'],
            'F': ['GM', 'TSLA'],
            'RIVN': ['TSLA', 'F'],
            'LCID': ['TSLA', 'RIVN'],
            
            # Tech Giants
            'AAPL': ['MSFT', 'GOOGL'],
            'MSFT': ['AAPL', 'GOOGL'],
            'GOOGL': ['MSFT', 'META'],
            'META': ['GOOGL', 'SNAP'],
            'AMZN': ['WMT', 'MSFT'],
            
            # Streaming
            'NFLX': ['DIS', 'PARA'],
            'DIS': ['NFLX', 'PARA'],
            
            # Semiconductors
            'NVDA': ['AMD', 'INTC'],
            'AMD': ['NVDA', 'INTC'],
            'INTC': ['AMD', 'NVDA'],
            
            # Retail
            'WMT': ['TGT', 'COST'],
            'TGT': ['WMT', 'COST'],
            'COST': ['WMT', 'TGT'],
            
            # Finance
            'JPM': ['BAC', 'WFC'],
            'BAC': ['JPM', 'C'],
            'WFC': ['JPM', 'BAC'],
            
            # Pharma
            'JNJ': ['PFE', 'ABBV'],
            'PFE': ['JNJ', 'MRK'],
            
            # Aerospace
            'BA': ['LMT', 'RTX'],
            'LMT': ['BA', 'RTX'],
            
            # Energy
            'XOM': ['CVX', 'COP'],
            'CVX': ['XOM', 'COP'],
            
            # Sportswear
            'NKE': ['ADDYY', 'UAA'],
            
            # Payment processors
            'V': ['MA', 'AXP'],
            'MA': ['V', 'AXP'],
        }
        
        # Check if we have a manual mapping
        if ticker in known_competitors:
            competitors = known_competitors[ticker]
            logger.info(f"Using manual mapping for {ticker}: {competitors}")
            return competitors
        
        # Otherwise, use sector-based fallback
        sector_competitors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Consumer Cyclical': ['HD', 'NKE', 'TGT', 'LOW'],  # Removed AMZN, TSLA to avoid confusion
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


class CompetitiveComparison:
    """
    Generate competitive analysis comparing a stock to AI-identified competitors.
    
    Shows:
    - Side-by-side statistics (like football league table)
    - Price performance charts (3M, 6M, 12M)
    - Key metrics comparison (P/E, dividend, market cap, etc.)
    """
    
    def __init__(
        self,
        ticker: str,
        api_key: Optional[str] = None,
        use_ollama: bool = False,
        manual_competitors: Optional[List[str]] = None
    ):
        """
        Initialize comparison.
        
        Args:
            ticker: Main stock ticker
            api_key: Groq API key for AI competitor detection
            use_ollama: If True, use Ollama (local) instead of Groq
            manual_competitors: Override AI with manual competitor list
        """
        self.ticker = ticker.upper()
        self.api_key = api_key
        self.use_ollama = use_ollama
        self.manual_competitors = manual_competitors
        self.competitors = []
        self.all_tickers = []
        self.data = {}
        self.competitor_reasoning = ""
    
    def detect_competitors(self) -> List[str]:
        """
        Detect competitors using AI or manual override.
        
        Returns:
            List of competitor tickers
        """
        if self.manual_competitors:
            self.competitors = [c.upper() for c in self.manual_competitors]
            logger.info(f"Using manual competitors: {self.competitors}")
            self.all_tickers = [self.ticker] + self.competitors
            return self.competitors
        
        # Get company info for AI detection
        try:
            ticker_obj = yf.Ticker(self.ticker)
            info = ticker_obj.info
            
            company_name = info.get('longName', self.ticker)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Use AI to detect competitors
            detector = CompetitorDetector(api_key=self.api_key, use_ollama=self.use_ollama)
            self.competitors = detector.find_competitors(
                ticker=self.ticker,
                company_name=company_name,
                sector=sector,
                industry=industry,
                num_competitors=2
            )
            
            self.all_tickers = [self.ticker] + self.competitors
            return self.competitors
            
        except Exception as e:
            logger.error(f"Failed to detect competitors: {e}")
            return []
    
    def fetch_all_data(self) -> Dict:
        """
        Fetch comprehensive data for all tickers.
        
        Returns:
            Dictionary with all comparison data
        """
        if not self.competitors:
            self.detect_competitors()
        
        if not self.competitors:
            logger.error("No competitors found")
            return {}
        
        logger.info(f"Fetching data for {self.ticker} and competitors: {self.competitors}")
        
        for ticker in self.all_tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Get price history for different periods
                hist_3m = ticker_obj.history(period='3mo')
                hist_6m = ticker_obj.history(period='6mo')
                hist_1y = ticker_obj.history(period='1y')
                
                self.data[ticker] = {
                    'info': info,
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'dividend_yield': info.get('dividendYield', 0),
                    'payout_ratio': info.get('payoutRatio'),
                    'profit_margin': info.get('profitMargins'),
                    'roe': info.get('returnOnEquity'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'beta': info.get('beta'),
                    '52w_high': info.get('fiftyTwoWeekHigh'),
                    '52w_low': info.get('fiftyTwoWeekLow'),
                    'avg_volume': info.get('averageVolume'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'company_name': info.get('longName', ticker),
                    # Price performance
                    'price_3m': hist_3m,
                    'price_6m': hist_6m,
                    'price_1y': hist_1y,
                    'return_3m': self._calculate_return(hist_3m),
                    'return_6m': self._calculate_return(hist_6m),
                    'return_1y': self._calculate_return(hist_1y),
                }
                
                logger.info(f"✓ Fetched data for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                self.data[ticker] = None
        
        return self.data
    
    def _calculate_return(self, hist_df: pd.DataFrame) -> Optional[float]:
        """Calculate percentage return from historical data."""
        if hist_df.empty or len(hist_df) < 2:
            return None
        
        try:
            start_price = hist_df['Close'].iloc[0]
            end_price = hist_df['Close'].iloc[-1]
            return ((end_price - start_price) / start_price) * 100
        except:
            return None
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table (like football statistics table).
        
        Returns:
            DataFrame with side-by-side comparison
        """
        if not self.data:
            self.fetch_all_data()
        
        comparison_data = []
        
        for ticker in self.all_tickers:
            data = self.data.get(ticker)
            if not data:
                continue
            
            comparison_data.append({
                'Company': data['company_name'],
                'Ticker': ticker,
                'Price': f"${data['current_price']:.2f}" if data['current_price'] else 'N/A',
                'Market Cap': f"${data['market_cap']/1e9:.2f}B" if data['market_cap'] else 'N/A',
                'P/E Ratio': f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else 'N/A',
                'Div Yield': f"{data['dividend_yield']*100:.2f}%" if data['dividend_yield'] else '0%',
                '3M Return': f"{data['return_3m']:+.1f}%" if data['return_3m'] else 'N/A',
                '6M Return': f"{data['return_6m']:+.1f}%" if data['return_6m'] else 'N/A',
                '1Y Return': f"{data['return_1y']:+.1f}%" if data['return_1y'] else 'N/A',
                'Profit Margin': f"{data['profit_margin']*100:.1f}%" if data['profit_margin'] else 'N/A',
                'ROE': f"{data['roe']*100:.1f}%" if data['roe'] else 'N/A',
                'Beta': f"{data['beta']:.2f}" if data['beta'] else 'N/A',
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def create_price_comparison_chart(self, period: str = '1y') -> go.Figure:
        """
        Create price comparison chart.
        
        Args:
            period: '3mo', '6mo', or '1y'
            
        Returns:
            Plotly figure
        """
        if not self.data:
            self.fetch_all_data()
        
        fig = go.Figure()
        
        # Map period to data key
        period_map = {
            '3mo': ('price_3m', '3 Months'),
            '6mo': ('price_6m', '6 Months'),
            '1y': ('price_1y', '1 Year')
        }
        
        data_key, title_period = period_map.get(period, ('price_1y', '1 Year'))
        
        # Add price lines for each ticker
        for ticker in self.all_tickers:
            data = self.data.get(ticker)
            if not data or data_key not in data:
                continue
            
            hist_df = data[data_key]
            if hist_df.empty:
                continue
            
            # Normalize to percentage change from start
            normalized = ((hist_df['Close'] / hist_df['Close'].iloc[0]) - 1) * 100
            
            is_main = ticker == self.ticker
            
            fig.add_trace(go.Scatter(
                x=hist_df.index,
                y=normalized,
                name=f"{ticker} ({data['company_name']})",
                line=dict(
                    width=3 if is_main else 2,
                    dash='solid' if is_main else 'dash'
                ),
                mode='lines'
            ))
        
        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"Price Performance Comparison - Last {title_period}",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_metrics_radar_chart(self) -> go.Figure:
        """
        Create radar chart comparing key metrics.
        
        Returns:
            Plotly figure
        """
        if not self.data:
            self.fetch_all_data()
        
        # Metrics to compare (normalized 0-100)
        metrics = ['P/E Ratio', 'ROE', 'Profit Margin', 'Revenue Growth', 'Div Yield']
        
        fig = go.Figure()
        
        for ticker in self.all_tickers:
            data = self.data.get(ticker)
            if not data:
                continue
            
            # Normalize metrics to 0-100 scale
            values = []
            
            # P/E (lower is better, invert scale)
            pe = data['pe_ratio']
            pe_norm = max(0, 100 - (pe / 50 * 100)) if pe and pe > 0 else 50
            values.append(pe_norm)
            
            # ROE (higher is better)
            roe = data['roe'] * 100 if data['roe'] else 0
            values.append(min(100, roe * 5))
            
            # Profit Margin (higher is better)
            margin = data['profit_margin'] * 100 if data['profit_margin'] else 0
            values.append(min(100, margin * 5))
            
            # Revenue Growth (higher is better)
            rev_growth = data['revenue_growth'] * 100 if data['revenue_growth'] else 0
            values.append(min(100, max(0, (rev_growth + 10) * 5)))
            
            # Dividend Yield (higher is better)
            div_yield = data['dividend_yield'] * 100 if data['dividend_yield'] else 0
            values.append(min(100, div_yield * 20))
            
            is_main = ticker == self.ticker
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself' if is_main else 'none',
                name=ticker,
                line=dict(width=3 if is_main else 2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Key Metrics Comparison",
            height=500
        )
        
        return fig
    
    def generate_summary(self) -> str:
        """
        Generate text summary of comparison.
        
        Returns:
            Summary text
        """
        if not self.data:
            self.fetch_all_data()
        
        main_data = self.data.get(self.ticker)
        if not main_data:
            return "Unable to generate summary - missing data"
        
        summary = f"**{main_data['company_name']} ({self.ticker})** Competitive Analysis\n\n"
        summary += f"**Sector**: {main_data['sector']} | **Industry**: {main_data['industry']}\n\n"
        summary += f"**Main Competitors**: {', '.join(self.competitors)}\n\n"
        
        # Price performance summary
        summary += "**Price Performance**:\n"
        if main_data['return_3m']:
            summary += f"- 3 Months: {main_data['return_3m']:+.1f}%\n"
        if main_data['return_6m']:
            summary += f"- 6 Months: {main_data['return_6m']:+.1f}%\n"
        if main_data['return_1y']:
            summary += f"- 1 Year: {main_data['return_1y']:+.1f}%\n"
        
        return summary