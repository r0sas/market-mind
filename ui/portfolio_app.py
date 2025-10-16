import streamlit as st
import pandas as pd
import plotly.express as px
from core.stock_portfolio import Stock_portfolio
import plotly.graph_objects as go
# Import UI components
from ui.components import AddStockForm, PortfolioTable, SummarySection, ChartSection, PlotSelectionDropdown

class PortfolioApp:
    def __init__(self):
        self.selected_stocks = []

        st.set_page_config(page_title="Stock ROI Tracker", layout="wide")
        st.title("📈 Stock ROI Tracker")

        if "portfolio" not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=[
                "Stock Symbol", "Shares", "Price", "Acquisition Date", "Current Price"
            ])

    def run(self):
        AddStockForm().render()
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
        ChartSection().render(selected)
        portfolio_rentability_data, stocks_rentability = Stock_portfolio().get_portfolio_historical_rentability(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_rentability_data["Date"], y=portfolio_rentability_data["Current Value"], mode='lines+markers', name='Current Valuation'))
        fig.add_trace(go.Scatter(x=portfolio_rentability_data["Date"], y=portfolio_rentability_data["Total Invested"], mode='lines+markers', name='Invested'))

        fig.update_layout(title="Portfolio", xaxis_title="Date", yaxis_title="Value (€)")

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(stocks_rentability, x="Date", y="ROI", color="Ticker", title="Stock ROI")
        st.plotly_chart(fig, use_container_width=True)