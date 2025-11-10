# streamlit_intrinsic_value_v3.py

import streamlit as st
import pandas as pd
import plotly.express as px

from core.oldscripts.DataFetcher import DataFetcher
from core.IV_Simplifier.IV_Simplifier import IVSimplifier
from core.oldscripts.ValuationCalculator import ValuationCalculator
from core.oldscripts.model_selector import ModelSelector

# ---- Sidebar ----
st.sidebar.title("Intrinsic Value Dashboard")

# Optional: Model selection
models = [
    "DCF Intrinsic Value",
    "DDM Single-Stage Value",
    "DDM Multi-Stage Value",
    "P/E Model Intrinsic Value",
    "Asset-Based Value",
    "Modern Graham Value"
]
selected_models = st.sidebar.multiselect("Select Models to Display", models, default=models)

# ---- Main Page ----
st.title("Intrinsic Value Calculator")

ticker_input = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT")

if st.button("Calculate Intrinsic Value"):
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    results = []
    margin_results = []

    for ticker in tickers:
        try:
            df_data = DataFetcher(ticker).get_comprehensive_data()
            df_iv = IVSimplifier(df_data).simplify()
            
            vc = ValuationCalculator(df_iv)
            vc.calculate_all_valuations()
            avg_value = vc.get_average_valuation()
            current_price = vc.current_price
            iv_values = vc.get_results()
            margin_analysis = vc.get_margin_of_safety()

            model_name_map = {
                'dcf': 'DCF Intrinsic Value',
                'ddm_single_stage': 'DDM Single-Stage Value',
                'ddm_multi_stage': 'DDM Multi-Stage Value',
                'pe_model': 'P/E Model Intrinsic Value',
                'asset_based': 'Asset-Based Value',
                'graham_value': 'Modern Graham Value'
            }

            # Prepare intrinsic value table
            
            iv_filtered = {"Ticker": ticker}
            iv_filtered["Current Share Price"] = current_price
            for key, name in model_name_map.items():
                if name in selected_models and key in iv_values:
                    iv_filtered[name] = iv_values[key]
            
            results.append(iv_filtered)

            # Prepare margin of safety table
            if margin_analysis:
                for model, data in margin_analysis.items():
                    if model_name_map.get(model, model) in selected_models:
                        margin_results.append({
                            "Ticker": ticker,
                            "Model": model_name_map.get(model, model),
                            "Current Price": data["current_price"],
                            "Target Buy Price": data["target_buy_price"],
                            "Intrinsic Value": data["intrinsic_value"],
                            "Margin of Safety (%)": round(data["margin_of_safety"] * 100, 1),
                            "Undervalued": data["is_undervalued"]
                            
                        })

        except Exception as e:
            st.error(f"Error calculating {ticker}: {e}")

    if results:
        # ---- Intrinsic Value Table ----
        df_iv_results = pd.DataFrame(results).set_index("Ticker")
        st.subheader("Intrinsic Value Table")
        st.dataframe(df_iv_results.style.format("${:,.2f}"))

        # ---- Intrinsic Value Plot ----
        st.subheader("Intrinsic Value Comparison")
        df_plot = df_iv_results.reset_index().melt(id_vars="Ticker", var_name="Model", value_name="Value")
        fig = px.bar(df_plot, x="Ticker", y="Value", color="Model", barmode="group",
                     title="Intrinsic Value by Model")
        st.plotly_chart(fig, use_container_width=True)

    if margin_results:
        # ---- Margin of Safety Table ----
        df_margin = pd.DataFrame(margin_results).set_index(["Ticker", "Model"])
        st.subheader("Margin of Safety Analysis")
        st.dataframe(df_margin.style.format({
            "Intrinsic Value": "${:,.2f}",
            "Current Price": "${:,.2f}",
            "Target Buy Price": "${:,.2f}"
        }))

        # ---- Margin of Safety Plot ----
        st.subheader("Margin of Safety Visualization")
        df_margin_plot = pd.DataFrame(margin_results)
        df_margin_plot["Color"] = df_margin_plot["Undervalued"].apply(lambda x: "green" if x else "red")

        fig2 = px.bar(
            df_margin_plot,
            x="Ticker",
            y="Margin of Safety (%)",
            color="Color",
            text="Margin of Safety (%)",
            facet_col="Model",
            color_discrete_map={"green": "green", "red": "red"},
            title="Margin of Safety by Model"
        )
        fig2.update_traces(textposition='outside')
        fig2.for_each_xaxis(lambda axis: axis.update(tickangle=45))
        st.plotly_chart(fig2, use_container_width=True)
