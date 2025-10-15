import streamlit as st


class PortfolioTable:
    """Displays and manages the editable portfolio table."""

    def render(self):
        if st.session_state.portfolio.empty:
            st.info("No acquisitions yet. Add one above to get started!")
            return None

        st.divider()
        st.subheader("💼 All Acquisitions")

        df = st.session_state.portfolio.copy()
        df["Investment (€)"] = df["Shares"] * df["Price"]
        df["Current Value (€)"] = df["Shares"] * df["Current Price"]
        df["ROI (%)"] = ((df["Current Price"] - df["Price"]) / df["Price"]) * 100

        df_edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

        if not df_edited.equals(st.session_state.portfolio):
            st.session_state.portfolio = df_edited.copy()
            st.success("✅ Portfolio updated after edit or removal.")

        return df_edited