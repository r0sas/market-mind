import streamlit as st
import pandas as pd
from core.stock_portfolio import Stock_portfolio

# Import UI components
from ui.components import AddStockForm, PortfolioTable, SummarySection, ChartSection

class PortfolioApp:
    def __init__(self):
        self.portfolio = Stock_portfolio()
        self.selected_stocks = []

        st.set_page_config(page_title="Stock ROI Tracker", layout="wide")
        st.title("📈 Stock ROI Tracker")

        if "portfolio" not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=[
                "Stock Symbol", "Shares", "Price", "Acquisition Date", "Current Price"
            ])

    def run(self):
        AddStockForm(self.portfolio).render()
        df = PortfolioTable().render()
        if df is None:
            return

        selected = SummarySection().render(df)

        if st.button("🗑️ Clear All Data"):
            st.session_state.portfolio = pd.DataFrame(columns=[
                "Stock Symbol", "Shares", "Price", "Acquisition Date", "Current Price"
            ])
            st.success("Portfolio cleared.")
        #PlotSelectionDropdown().render()
        ChartSection(self.portfolio).render(selected)
