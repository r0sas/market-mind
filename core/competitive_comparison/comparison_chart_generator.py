"""
comparison_chart_generator.py

Create Plotly figures: price comparison and radar chart of metrics.
"""

from typing import Dict, List
import logging
import plotly.graph_objects as go
import pandas as pd

logger = logging.getLogger(__name__)


class ComparisonChartGenerator:
    def __init__(self, data: Dict[str, Dict], main_ticker: str):
        self.data = data
        self.main_ticker = main_ticker

    def create_price_comparison_chart(self, period: str = '1y') -> go.Figure:
        """
        Create normalized price-return comparison chart.
        period: '3mo', '6mo', '1y'
        """
        period_map = {'3mo': 'price_3m', '6mo': 'price_6m', '1y': 'price_1y'}
        data_key = period_map.get(period, 'price_1y')
        fig = go.Figure()

        for ticker, d in (self.data or {}).items():
            if not d:
                continue
            hist = d.get(data_key)
            if hist is None or hist.empty:
                continue
            try:
                normalized = ((hist['Close'] / hist['Close'].iloc[0]) - 1) * 100
                is_main = ticker == self.main_ticker
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    name=f"{ticker} ({d.get('company_name','')})",
                    mode='lines',
                    line=dict(width=3 if is_main else 2, dash='solid' if is_main else 'dash')
                ))
            except Exception:
                logger.debug(f"Skipping {ticker} for price chart due to history parse error")

        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"Price Performance Comparison - Last {period.upper()}",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        logger.info("Price comparison chart created")
        return fig

    def create_metrics_radar_chart(self) -> go.Figure:
        """
        Radar chart comparing normalized P/E, ROE, Profit Margin, Revenue Growth, Div Yield.
        """
        metrics = ['P/E Ratio', 'ROE', 'Profit Margin', 'Revenue Growth', 'Div Yield']
        fig = go.Figure()

        for ticker, d in (self.data or {}).items():
            if not d:
                continue
            # prepare normalized values
            pe = d.get('pe_ratio') or None
            pe_norm = max(0, 100 - (pe / 50 * 100)) if pe and pe > 0 else 50

            roe = (d.get('roe') or 0) * 100
            roe_norm = min(100, roe * 5)

            margin = (d.get('profit_margin') or 0) * 100
            margin_norm = min(100, margin * 5)

            rev_g = (d.get('revenue_growth') or 0) * 100
            rev_norm = min(100, max(0, (rev_g + 10) * 5))

            div_y = (d.get('dividend_yield') or 0) * 100
            div_norm = min(100, div_y * 20)

            values = [pe_norm, roe_norm, margin_norm, rev_norm, div_norm]
            is_main = ticker == self.main_ticker

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself' if is_main else 'none',
                name=ticker,
                line=dict(width=3 if is_main else 2)
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Key Metrics Comparison",
            height=500,
            showlegend=True
        )
        logger.info("Metrics radar chart created")
        return fig
