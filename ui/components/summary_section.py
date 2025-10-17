import streamlit as st


class SummarySection:
    """Displays aggregated ROI summary by stock."""

    def render(self, df):
        st.divider()
        st.subheader("📊 ROI Summary by Stock Symbol")

        summary = (
            df.groupby("Ticker")
            .agg({
                "Investment ($)": "sum",
                "Current Value ($)": "sum"
            })
            .reset_index()
        )
        summary["ROI (%)"] = (
            (summary["Current Value ($)"] - summary["Investment ($)"]) /
            summary["Investment ($)"]
        ) * 100

        st.dataframe(summary)