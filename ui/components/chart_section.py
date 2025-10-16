import streamlit as st
import plotly.express as px
from core.stock_portfolio import Stock_portfolio

class ChartSection:
    """Plots historical data for selected stocks."""

    def __init__(self):
        pass

    def render(self, selected_stocks):
        if not selected_stocks:
            st.info("Select one or more stocks to see historical data.")
            return

        st.subheader("📈 Historical Data Plot")
        hist_data = Stock_portfolio().get_historical_data(selected_stocks)
        fig = px.line(hist_data, x="Date", y="Close", color="Ticker", title="Historical Prices")
        st.plotly_chart(fig, use_container_width=True)