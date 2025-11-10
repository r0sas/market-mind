"""
Main orchestrator: CompetitiveComparison

Combines competitor detection, data fetching, table generation and charting.
"""

from typing import List, Dict, Optional
import logging

from .competitor_detector import CompetitorDetector
from .comparison_data_fetcher import ComparisonDataFetcher
from .comparison_table_generator import ComparisonTableGenerator
from .comparison_chart_generator import ComparisonChartGenerator

logger = logging.getLogger(__name__)


class CompetitiveComparison:
    """
    High-level class to coordinate competitor detection and comparisons.
    """

    def __init__(
        self,
        ticker: str,
        api_key: Optional[str] = None,
        use_ollama: bool = False,
        manual_competitors: Optional[List[str]] = None,
    ):
        self.ticker = ticker.upper()
        self.api_key = api_key
        self.use_ollama = use_ollama
        self.manual_competitors = [m.upper() for m in manual_competitors] if manual_competitors else None

        self.competitors: List[str] = []
        self.all_tickers: List[str] = []
        self.data: Dict[str, Dict] = {}
        self._detector = CompetitorDetector(api_key=api_key, use_ollama=use_ollama)

    # --------------------
    # High-level workflow
    # --------------------
    def detect_competitors(self) -> List[str]:
        if self.manual_competitors:
            self.competitors = self.manual_competitors[:2]
            logger.info(f"Using manual competitors: {self.competitors}")
            self.all_tickers = [self.ticker] + self.competitors
            return self.competitors

        try:
            ticker_obj = self._get_yf_ticker(self.ticker)
            info = ticker_obj.info
            company_name = info.get('longName', self.ticker)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            self.competitors = self._detector.find_competitors(
                ticker=self.ticker,
                company_name=company_name,
                sector=sector,
                industry=industry,
                num_competitors=2,
            )
            self.all_tickers = [self.ticker] + self.competitors
            return self.competitors
        except Exception as e:
            logger.error(f"Failed to detect competitors: {e}")
            return []

    def fetch_all_data(self) -> Dict[str, Dict]:
        if not self.competitors:
            self.detect_competitors()

        if not self.competitors:
            logger.error("No competitors available for fetching")
            return {}

        fetcher = ComparisonDataFetcher([self.ticker] + self.competitors)
        self.data = fetcher.fetch_all()
        return self.data

    def generate_comparison_table(self):
        if not self.data:
            self.fetch_all_data()
        generator = ComparisonTableGenerator(self.data)
        return generator.build_table()

    def create_price_comparison_chart(self, period: str = '1y'):
        if not self.data:
            self.fetch_all_data()
        charter = ComparisonChartGenerator(self.data, main_ticker=self.ticker)
        return charter.create_price_comparison_chart(period=period)

    def create_metrics_radar_chart(self):
        if not self.data:
            self.fetch_all_data()
        charter = ComparisonChartGenerator(self.data, main_ticker=self.ticker)
        return charter.create_metrics_radar_chart()

    def generate_summary(self) -> str:
        if not self.data:
            self.fetch_all_data()
        main = self.data.get(self.ticker)
        if not main:
            return "Unable to generate summary - missing data"
        summary = f"**{main.get('company_name', self.ticker)} ({self.ticker})** Competitive Analysis\n\n"
        summary += f"**Sector**: {main.get('sector')} | **Industry**: {main.get('industry')}\n\n"
        summary += f"**Main Competitors**: {', '.join(self.competitors)}\n\n"
        summary += "**Price Performance**:\n"
        if main.get('return_3m') is not None:
            summary += f"- 3 Months: {main['return_3m']:+.1f}%\n"
        if main.get('return_6m') is not None:
            summary += f"- 6 Months: {main['return_6m']:+.1f}%\n"
        if main.get('return_1y') is not None:
            summary += f"- 1 Year: {main['return_1y']:+.1f}%\n"
        return summary

    # --------------------
    # Small internal helpers
    # --------------------
    @staticmethod
    def _get_yf_ticker(ticker: str):
        import yfinance as yf
        return yf.Ticker(ticker)
