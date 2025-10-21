import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict

class SummarySection:
    """Displays aggregated ROI summary by stock."""
    def render(self, summary, sharpe, drawdown, positions_df):
        st.title("📊 Portfolio Performance Report")
        st.markdown("---")

        # Quick metrics summary at the top
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Investment:", f"{summary['total_investment']:,.2f}$")
        col2.metric("Current Value", f"{summary['current_value']:,.2f}$")
        col4.metric("Total Profit/Loss", f"{summary['total_profit_loss']:,.2f}$")
        col3.metric("Total Return", f"{summary['total_return_pct']:.2f}%")

        # Overview
        with st.expander("📘 Overview", expanded=True):
            st.markdown(f"""
            **Period:** {summary['start_date']} → {summary['end_date']}  
            **Days Invested:** {summary['days_invested']}  
            **Number of Positions:** {summary['num_positions']}
            """)

        # Performance
        with st.expander("💰 Performance Metrics", expanded=False):
            st.markdown(f"""
            **Current Value:** {summary['current_value']:,.2f}\$  
            **Total Investment:** {summary['total_investment']:,.2f}\$  
            **Total Profit/Loss:** {summary['total_profit_loss']:,.2f}\$  
            **Total Return:** {summary['total_return_pct']:.2f}%  
            **Annualized Return:** {summary['annualized_return_pct']:.2f}%  
            **Total Dividends:** {summary['total_dividends']:,.2f}$
            """)

        # Risk
        with st.expander("⚖️ Risk Metrics", expanded=False):
            st.markdown(f"""
            **Sharpe Ratio:** {sharpe:.4f}  
            **Max Drawdown:** {drawdown['max_drawdown_pct']:.2f}%  
            **Peak Date:** {drawdown['peak_date']}  
            **Trough Date:** {drawdown['trough_date']}  
            **Recovery Date:** {drawdown['recovery_date'] or 'Not Recovered'}
            """)

        # Positions Summary
        with st.expander("📈 Positions Summary", expanded=True):
            st.markdown("### Current Holdings Overview")
            styled_df = positions_df.style.format({
                "Investment ($)": "${:,.2f}",
                "Current Value": "${:,.2f}",
                "Profit/Loss": "${:,.2f}",
                "ROI (%)": "{:.2f}%",
                "Cumulative Dividends": "${:,.2f}"
            })

            st.dataframe(styled_df, use_container_width=True)

    def render_allocation_charts(self, data: Dict[str, pd.DataFrame]):
        with st.expander("📊 Portfolio Allocation by Sector and Industry"):
            # Two columns side by side
            col1, col2 = st.columns(2)

            # ---- Sector Pie Chart ----
            with col1:
                st.markdown("🏢 **By Sector**")
                df_sector = data["sector"]
                fig_sector = px.pie(
                    df_sector,
                    names="Sector",
                    values="Portfolio Weight (%)",
                    hover_data={"Current Value": ":,.2f", "Portfolio Weight (%)": ":.2f"},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_sector.update_traces(
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Value: %{customdata[0][0]:,.2f}$<br>Weight: %{value:.2f}%"
                )
                fig_sector.update_layout(
                    legend_title="Sector",
                    showlegend=True,
                    height=400,
                    margin=dict(t=40, b=20)
                )
                st.plotly_chart(fig_sector, use_container_width=True)

            # ---- Industry Pie Chart ----
            with col2:
                st.markdown("🏭 **By Industry**")
                df_industry = data["industry"]
                fig_industry = px.pie(
                    df_industry,
                    names="Industry",
                    values="Portfolio Weight (%)",
                    hover_data={"Current Value": ":,.2f", "Portfolio Weight (%)": ":.2f"},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_industry.update_traces(
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Value: %{customdata[0][0]:,.2f}$<br>Weight: %{value:.2f}%"
                )
                fig_industry.update_layout(
                    legend_title="Industry",
                    showlegend=True,
                    height=400,
                    margin=dict(t=40, b=20)
                )
                st.plotly_chart(fig_industry, use_container_width=True)
        st.markdown("---")
        st.caption("Generated automatically — Portfolio Performance Report © 2025")