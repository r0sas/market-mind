import streamlit as st


class SummarySection:
    """Displays aggregated ROI summary by stock."""

    def render(self, df):
        st.divider()
        st.subheader("📊 ROI Summary by Stock Symbol")

        summary = (
            df.groupby("Stock Symbol")
            .agg({
                "Investment (€)": "sum",
                "Current Value (€)": "sum"
            })
            .reset_index()
        )
        summary["ROI (%)"] = (
            (summary["Current Value (€)"] - summary["Investment (€)"]) /
            summary["Investment (€)"]
        ) * 100

        summary["Select"] = [False for _ in range(summary.shape[0])]
        summary = st.data_editor(summary, use_container_width=True)
        return summary[summary["Select"] == True]["Stock Symbol"].tolist()