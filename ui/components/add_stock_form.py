import streamlit as st
import pandas as pd
from datetime import date
from core.stock_calculator import StockCalculator
from typing import Optional


class AddStockForm:
    """Handles user input for adding new stock acquisitions."""

    def __init__(self, calculator: Optional[StockCalculator] = None):
            """
            Initialize the AddStockForm with a stock calculator.
            
            Args:
                calculator: Optional StockCalculator instance. Creates new one if not provided.
            """
            self._calculator = calculator or StockCalculator()
        

    def render(self):
        st.subheader("➕ Add a New Acquisition")
        with st.form("add_stock_form", clear_on_submit=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                stock_symbol = st.text_input("Stock Symbol (e.g., TSM)").upper()
            with col2:
                shares = st.number_input("Shares", min_value=0.0, step=1.0)
            with col3:
                price = st.number_input("Acquisition Price ($)", min_value=0.0, step=0.01)
            with col4:
                acquisition_date = st.date_input("Date", value=date.today())

            submitted = st.form_submit_button("Add Acquisition")

        if submitted:
            if stock_symbol and shares > 0 and price > 0:
                self._add_to_portfolio(stock_symbol, shares, price, acquisition_date)
            else:
                st.error("Please fill all fields correctly.")

    def _add_to_portfolio(self, stock_symbol, shares, price, acquisition_date):
        current_price = self._calculator.get_current_prices([stock_symbol])[stock_symbol]

        new_row = {
            "Ticker": stock_symbol,
            "Shares": shares,
            "Price": price,
            "Date": acquisition_date,
            "Current Price": current_price
        }

        st.session_state.portfolio = pd.concat(
            [st.session_state.portfolio, pd.DataFrame([new_row])],
            ignore_index=True
        )
        st.success(f"Added {shares} shares of {stock_symbol} ✅")