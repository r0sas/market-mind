import streamlit as st
import pandas as pd
import io

class PortfolioButtonsSection:
    def __init__(self):
        pass

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
        st.session_state.portfolio = pd.read_csv(uploaded_file)