import streamlit as st
import pandas as pd
from datetime import date


class AddStockForm:
    """Handles user input for adding new stock acquisitions."""

    def __init__(self, portfolio):
        self.portfolio = portfolio

    def render(self):
        st.subheader("➕ Add a New Acquisition")
        with st.form("add_stock_form", clear_on_submit=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                stock_symbol = st.text_input("Stock Symbol (e.g., TSM)").upper()
            with col2:
                shares = st.number_input("Shares", min_value=0.0, step=1.0)
            with col3:
                price = st.number_input("Acquisition Price (€)", min_value=0.0, step=0.01)
            with col4:
                acquisition_date = st.date_input("Acquisition Date", value=date.today())

            submitted = st.form_submit_button("Add Acquisition")

        if submitted:
            if stock_symbol and shares > 0 and price > 0:
                self._add_to_portfolio(stock_symbol, shares, price, acquisition_date)
            else:
                st.error("Please fill all fields correctly.")

    def _add_to_portfolio(self, stock_symbol, shares, price, acquisition_date):
        self.portfolio.add_position(stock_symbol, shares, price, acquisition_date)
        stock_obj = self.portfolio.get_stock_object(stock_symbol)
        current_price = stock_obj.fetch_current_price()

        new_row = {
            "Stock Symbol": stock_symbol,
            "Shares": shares,
            "Price": price,
            "Acquisition Date": acquisition_date,
            "Current Price": current_price
        }

        st.session_state.portfolio = pd.concat(
            [st.session_state.portfolio, pd.DataFrame([new_row])],
            ignore_index=True
        )
        st.success(f"Added {shares} shares of {stock_symbol} ✅")