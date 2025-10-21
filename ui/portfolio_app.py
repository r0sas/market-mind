import streamlit as st
import pandas as pd
# Import UI components
from ui.components import AddStockForm, PortfolioTable, SummarySection, ChartSection, PortfolioButtonsSection
from core.portfolio_analyzer import PortfolioAnalyzer
import io

class PortfolioApp:
    def __init__(self):
        self.selected_stocks = []
        self.analyzer = PortfolioAnalyzer()

        st.set_page_config(page_title="Stock ROI Tracker", layout="wide")
        st.title("📈 Stock ROI Tracker")

        if "portfolio" not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=[
                "Ticker", "Shares", "Price", "Date", "Current Price"
            ])

    def run(self):
        AddStockForm().render()
        # Create CSV buffer for saving
        PortfolioButtonsSection().render()
        df = PortfolioTable().render()
        if df is None:
            return
        portfolio = self.analyzer.analyze_portfolio(df)
        pos_summary = self.analyzer.get_position_summary(portfolio)
        portfolio_summary = self.analyzer.get_portfolio_summary(portfolio)
        drawdown = self.analyzer.calculate_max_drawdown(portfolio)
        sharpe = self.analyzer.calculate_sharpe_ratio(portfolio)
        SummarySection().render(portfolio_summary, sharpe, drawdown, pos_summary)
        allocations = self.analyzer.get_latest_allocations(portfolio)
        SummarySection().render_allocation_charts(allocations)
        ChartSection().render(portfolio["Ticker"].unique().tolist(), portfolio)
        print(self.analyzer.get_latest_allocations(portfolio))