"""
comparison_data_fetcher.py

Fetches financial info and price history for a list of tickers using yfinance.
"""

from typing import Dict, List, Optional
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class ComparisonDataFetcher:
    """
    Fetches detailed info and historical price data for tickers.
    """

    def __init__(self, tickers: List[str]):
        self.tickers = [t.upper() for t in tickers]
        self.data: Dict[str, Optional[Dict]] = {}

    def fetch_all(self) -> Dict[str, Optional[Dict]]:
        """
        Fetch info and history for all tickers. Returns dict keyed by ticker.
        """
        for ticker in self.tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info

                hist_3m = t.history(period='3mo')
                hist_6m = t.history(period='6mo')
                hist_1y = t.history(period='1y')

                item = {
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
                    'price_3m': hist_3m,
                    'price_6m': hist_6m,
                    'price_1y': hist_1y,
                }

                # calculate returns
                item['return_3m'] = self._calculate_return(hist_3m)
                item['return_6m'] = self._calculate_return(hist_6m)
                item['return_1y'] = self._calculate_return(hist_1y)

                self.data[ticker] = item
                logger.info(f"✓ Fetched data for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                self.data[ticker] = None

        return self.data

    @staticmethod
    def _calculate_return(hist_df: pd.DataFrame) -> Optional[float]:
        """Return percentage from first close to last close in DataFrame."""
        if hist_df is None or hist_df.empty or 'Close' not in hist_df.columns:
            return None
        try:
            start = hist_df['Close'].iloc[0]
            end = hist_df['Close'].iloc[-1]
            return ((end - start) / start) * 100
        except Exception:
            return None
