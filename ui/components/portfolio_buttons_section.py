import streamlit as st
import pandas as pd
import io
from core.stock_calculator import StockCalculator
from typing import Optional

class PortfolioButtonsSection:
    def __init__(self, calculator: Optional[StockCalculator] = None):
            """
            Initialize the AddStockForm with a stock calculator.
            
            Args:
                calculator: Optional StockCalculator instance. Creates new one if not provided.
            """
            self._calculator = calculator or StockCalculator()
    def render(self):
        # --- BUTTONS SIDE BY SIDE ---
        col1, col2 = st.columns(2)

        # Prepare CSV for download
        csv_buffer = io.BytesIO()
        st.session_state.portfolio.to_csv(csv_buffer, index=False)

        with col1:
            if st.button("🗑️ Clear All Data"):
                self.clear_portfolio()
                st.success("Portfolio cleared.")

        with col2:
            st.download_button(
                label="💾 Save Portfolio",
                data=csv_buffer.getvalue(),
                file_name="portfolio.csv",
                mime="text/csv"
            )

        # --- LOAD SECTION BELOW ---
        uploaded_file = st.file_uploader("📂 Load saved DataFrame", type=["csv"])
        if uploaded_file is not None:
            self.load_portfolio(uploaded_file)
            st.success("✅ DataFrame loaded successfully!")
        

    def clear_portfolio(self):
        """Clears the portfolio DataFrame."""
        st.session_state.portfolio = pd.DataFrame(columns=[
            "Ticker", "Shares", "Price", "Date", "Current Price"
        ])

    def load_portfolio(self, uploaded_file):
        """Loads a CSV file into the portfolio DataFrame."""
        loaded_df = pd.read_csv(uploaded_file)
        all_tickers = loaded_df["Ticker"].unique().tolist()
        price_dict = self._calculator.get_current_prices(all_tickers)
        loaded_df.loc[:, "Current Price"] = loaded_df["Ticker"].map(price_dict)
        st.session_state.portfolio = loaded_df