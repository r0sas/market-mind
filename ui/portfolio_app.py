import streamlit as st
import pandas as pd
from core.portfolio_manager import PortfolioManager
# Import UI components
from ui.components import AddStockForm, PortfolioTable, SummarySection, ChartSection, PortfolioButtonsSection
import io

class PortfolioApp:
    def __init__(self):
        self.selected_stocks = []
        self.portfolio_manager = PortfolioManager()

        st.set_page_config(page_title="Stock ROI Tracker", layout="wide")
        st.title("📈 Stock ROI Tracker")

        if "portfolio" not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=[
                "Ticker", "Shares", "Price", "Date", "Current Price"
            ])

    def run(self):
        AddStockForm(self.portfolio_manager).render()
        # Create CSV buffer for saving
        PortfolioButtonsSection().render()
        df = PortfolioTable().render()
        if df is None:
            return
        SummarySection().render(df)
        # portfolio_rentability_data, stocks_history = Stock_portfolio().get_portfolio_historical_rentability(df)
        stocks_history = self.portfolio_manager.get_portfolio_history(df)
        print(stocks_history)
        portfolio_rentability_data = self.portfolio_manager.get_portfolio_rentability(stocks_history)
        print("Portfolio Rentability")
        print(portfolio_rentability_data)
        stocks_rentability = self.portfolio_manager.get_stocks_rentability(stocks_history)
        final_df = pd.concat([stocks_rentability, portfolio_rentability_data])
        ChartSection(self.portfolio_manager).render(final_df["Ticker"].unique().tolist(), final_df)