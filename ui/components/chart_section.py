import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class ChartSection:
    """Plots historical data for selected stocks."""

    def __init__(self):
        self.metrics_list = ["close", "Current Value", "Investment ($)", "dividends", "Cumulative Dividends", "Profit/Loss", "Rentability", "Cumulative Rentability", "ROI"]

    def render(self, selected_stocks: list, df):
        if not selected_stocks:
            st.info("Select one or more stocks to see historical data.")
            return

        st.subheader("📈 Historical Data Plot")
        # fig = px.line(hist_data, x="Date", y="Close", color="Ticker", title="Historical Prices")
        # st.plotly_chart(fig, use_container_width=True)

        # --- User multi-select ---
        tickers = selected_stocks
        selected_tickers = st.multiselect("Select tickers to compare", tickers, default=[tickers[0]])
        selected_metrics = st.multiselect("Metrics", self.metrics_list, default=self.metrics_list[0])
        # --- Build figure ---
        fig = go.Figure()

        fig = px.line(
            df[df['Ticker'].isin(selected_tickers)],
            x='Date',
            y=selected_metrics,
            color='Ticker',
            title=f"Comparison: {', '.join(selected_metrics)}"
        )

        # --- Display in Streamlit ---
        st.plotly_chart(fig, use_container_width=True)