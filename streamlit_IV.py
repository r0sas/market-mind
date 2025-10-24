import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import time

from core.DataFetcher import DataFetcher, DataFetcherError
from core.IVSimplifier import IVSimplifier, SimplifierError
from core.ValuationCalculator import ValuationCalculator, ValuationError
from core.model_selector import ModelSelector
from core.Config import (
    MODEL_DISPLAY_NAMES,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_MARGIN_OF_SAFETY,
    MIN_HISTORICAL_YEARS
)

# Page configuration
st.set_page_config(
    page_title="Intrinsic Value Calculator",
    page_icon="📈",
    layout="wide"
)

# Cache data fetching to avoid repeated API calls
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data with caching (1 hour TTL)"""
    fetcher = DataFetcher(ticker)
    return fetcher.get_comprehensive_data()


def create_confidence_badge(confidence: str) -> str:
    """Create HTML badge for confidence level"""
    colors = {
        'High': '#28a745',
        'Medium': '#ffc107', 
        'Low': '#dc3545',
        'N/A': '#6c757d'
    }
    color = colors.get(confidence, colors['N/A'])
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:3px;font-size:0.8em">{confidence}</span>'


def display_data_quality_warnings(warnings: List[str]):
    """Display data quality warnings"""
    if warnings:
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)


def create_sensitivity_plot(sensitivity_data: Dict, model_name: str):
    """Create sensitivity analysis plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sensitivity_data['values'],
        y=sensitivity_data['valuations'],
        mode='lines+markers',
        name=model_name,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"Sensitivity Analysis: {model_name}",
        xaxis_title=sensitivity_data['parameter'].replace('_', ' ').title(),
        yaxis_title="Intrinsic Value ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


# ---- Sidebar Configuration ----
st.sidebar.title("📊 Intrinsic Value Dashboard")
st.sidebar.markdown("---")

st.sidebar.subheader("Model Selection")
all_models = list(MODEL_DISPLAY_NAMES.values())
selected_models = st.sidebar.multiselect(
    "Select Valuation Models",
    all_models,
    default=all_models,
    help="Choose which valuation models to calculate and display"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Parameters")

with st.sidebar.expander("⚙️ DCF Parameters"):
    discount_rate = st.slider(
        "Discount Rate (%)",
        min_value=5.0,
        max_value=20.0,
        value=DEFAULT_DISCOUNT_RATE * 100,
        step=0.5,
        help="Required rate of return (WACC)"
    ) / 100
    
    terminal_growth = st.slider(
        "Terminal Growth Rate (%)",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_TERMINAL_GROWTH * 100,
        step=0.25,
        help="Perpetual growth rate for terminal value"
    ) / 100

with st.sidebar.expander("📈 Analysis Options"):
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_warnings = st.checkbox("Show Model Warnings", value=True)
    use_weighted_avg = st.checkbox(
        "Use Weighted Average",
        value=False,
        help="Weight models by confidence score"
    )
    
    margin_of_safety_pct = st.slider(
        "Target Margin of Safety (%)",
        min_value=10,
        max_value=50,
        value=int(DEFAULT_MARGIN_OF_SAFETY * 100),
        step=5,
        help="Desired discount from intrinsic value"
    ) / 100

with st.sidebar.expander("🔍 Sensitivity Analysis"):
    enable_sensitivity = st.checkbox("Enable Sensitivity Analysis", value=False)
    if enable_sensitivity:
        sensitivity_model = st.selectbox(
            "Model to Analyze",
            ["DCF", "DDM Multi-Stage"]
        )
        sensitivity_param = st.selectbox(
            "Parameter to Vary",
            ["Discount Rate", "Terminal Growth", "Growth Rate"]
        )

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: Enter multiple tickers separated by commas to compare stocks"
)

# ---- Main Page ----
st.title("📈 Intrinsic Value Calculator")
st.markdown(
    "Calculate intrinsic value using multiple valuation models: "
    "DCF, DDM, P/E Model, Asset-Based, and Graham Formula"
)

# Ticker input
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "Enter Stock Tickers",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter one or more ticker symbols separated by commas"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    calculate_button = st.button("🔍 Calculate", type="primary", use_container_width=True)

if calculate_button:
    # Validate input
    if not ticker_input.strip():
        st.error("❌ Please enter at least one ticker symbol")
        st.stop()
    
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    
    # Remove duplicates
    tickers = list(dict.fromkeys(tickers))
    
    st.markdown("---")
    
    # Initialize results storage
    results = []
    margin_results = []
    all_warnings = {}
    failed_tickers = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Processing {ticker}... ({idx + 1}/{len(tickers)})")
        
        try:
            # Fetch data
            with st.spinner(f"Fetching data for {ticker}..."):
                df_data = fetch_stock_data(ticker)
            
            # Simplify data
            simplifier = IVSimplifier(df_data)
            df_iv = simplifier.simplify()
            
            # Get data quality info
            quality_report = simplifier.get_data_quality_report()
            all_warnings[ticker] = quality_report.get('warnings', [])
            
            # Calculate valuations
            vc = ValuationCalculator(df_iv)
            
            # Calculate each model with appropriate parameters
            vc.calculate_dcf(
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth
            )
            vc.calculate_ddm(
                required_rate=discount_rate,
                terminal_growth=terminal_growth
            )
            vc.calculate_pe_model()
            vc.calculate_asset_based()
            vc.calculate_graham_value()
            
            avg_value = vc.get_average_valuation(weighted=use_weighted_avg)
            current_price = vc.current_price
            iv_values = vc.get_results()
            confidence_scores = vc.get_confidence_scores()
            model_warnings = vc.get_model_warnings()
            margin_analysis = vc.get_margin_of_safety(target_margin=margin_of_safety_pct)
            
            # Reverse model name mapping
            model_name_map = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
            
            # Prepare intrinsic value results
            iv_filtered = {
                "Ticker": ticker,
                "Current Price": current_price,
                "Years of Data": quality_report.get('num_years', 'N/A')
            }
            
            for display_name in selected_models:
                key = model_name_map.get(display_name)
                if key and key in iv_values:
                    iv_filtered[display_name] = iv_values[key]
                    if show_confidence and key in confidence_scores:
                        iv_filtered[f"{display_name}_Confidence"] = confidence_scores[key]
            
            # Add average
            if avg_value:
                avg_label = "Weighted Average" if use_weighted_avg else "Average"
                iv_filtered[avg_label] = avg_value
            
            results.append(iv_filtered)
            
            # Prepare margin of safety results
            if margin_analysis:
                for model, data in margin_analysis.items():
                    display_name = MODEL_DISPLAY_NAMES.get(model, model)
                    if display_name in selected_models:
                        margin_results.append({
                            "Ticker": ticker,
                            "Model": display_name,
                            "Intrinsic Value": data["intrinsic_value"],
                            "Current Price": data["current_price"],
                            "Margin of Safety (%)": round(data["margin_of_safety"] * 100, 1),
                            "Target Buy Price": data["target_buy_price"],
                            "Status": "✓ Undervalued" if data["is_undervalued"] else "✗ Overvalued",
                            "Confidence": data.get("confidence", "N/A")
                        })
        
        except (DataFetcherError, SimplifierError, ValuationError) as e:
            st.error(f"❌ Error processing {ticker}: {str(e)}")
            failed_tickers.append(ticker)
        except Exception as e:
            st.error(f"❌ Unexpected error with {ticker}: {str(e)}")
            failed_tickers.append(ticker)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(tickers))
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if results:
        st.success(f"✅ Successfully analyzed {len(results)} stock(s)")
        
        # Display warnings if enabled
        if show_warnings and any(all_warnings.values()):
            st.markdown("### ⚠️ Data Quality Warnings")
            for ticker, warnings in all_warnings.items():
                if warnings:
                    with st.expander(f"{ticker} - {len(warnings)} warning(s)"):
                        for warning in warnings:
                            st.warning(warning)
        
        st.markdown("---")
        
        # ---- Intrinsic Value Table ----
        st.markdown("### 💰 Intrinsic Value Summary")
        
        df_iv_results = pd.DataFrame(results)
        
        # Separate confidence columns if present
        value_cols = [col for col in df_iv_results.columns if not col.endswith('_Confidence')]
        confidence_cols = [col for col in df_iv_results.columns if col.endswith('_Confidence')]
        
        # Format the dataframe
        df_display = df_iv_results[value_cols].set_index("Ticker")
        
        # Create formatted version for display
        format_dict = {col: "${:,.2f}" for col in df_display.columns if col not in ["Years of Data"]}
        
        st.dataframe(
            df_display.style.format(format_dict).background_gradient(
                subset=[col for col in df_display.columns if col != "Years of Data"],
                cmap="RdYlGn",
                axis=1
            ),
            use_container_width=True
        )
        
        # Show confidence scores in separate table if enabled
        if show_confidence and confidence_cols:
            with st.expander("📊 Model Confidence Scores"):
                confidence_data = []
                for _, row in df_iv_results.iterrows():
                    conf_row = {"Ticker": row["Ticker"]}
                    for conf_col in confidence_cols:
                        model_name = conf_col.replace("_Confidence", "")
                        conf_row[model_name] = row[conf_col]
                    confidence_data.append(conf_row)
                
                df_confidence = pd.DataFrame(confidence_data).set_index("Ticker")
                
                # Style confidence scores with colors
                def color_confidence(val):
                    colors = {'High': 'background-color: #d4edda', 
                             'Medium': 'background-color: #fff3cd',
                             'Low': 'background-color: #f8d7da'}
                    return colors.get(val, '')
                
                st.dataframe(
                    df_confidence.style.applymap(color_confidence),
                    use_container_width=True
                )
        
        # ---- Visualization ----
        st.markdown("### 📊 Intrinsic Value Comparison")
        
        # Prepare data for plotting
        plot_cols = [col for col in df_display.columns if col not in ["Years of Data", "Current Price"]]
        df_plot = df_iv_results[["Ticker"] + plot_cols].melt(
            id_vars="Ticker",
            var_name="Model",
            value_name="Value"
        )
        
        fig = px.bar(
            df_plot,
            x="Ticker",
            y="Value",
            color="Model",
            barmode="group",
            title="Intrinsic Value by Model",
            labels={"Value": "Value ($)", "Model": "Valuation Model"}
        )
        
        # Add current price line if available
        if "Current Price" in df_display.columns:
            for ticker in df_display.index:
                current = df_display.loc[ticker, "Current Price"]
                fig.add_hline(
                    y=current,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{ticker} Current: ${current:.2f}",
                    annotation_position="right"
                )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ---- Margin of Safety Analysis ----
        if margin_results:
            st.markdown("### 🎯 Margin of Safety Analysis")
            
            df_margin = pd.DataFrame(margin_results)
            
            # Display table
            df_margin_display = df_margin.set_index(["Ticker", "Model"])
            
            format_dict_margin = {
                "Intrinsic Value": "${:,.2f}",
                "Current Price": "${:,.2f}",
                "Target Buy Price": "${:,.2f}"
            }
            
            st.dataframe(
                df_margin_display.style.format(format_dict_margin).applymap(
                    lambda x: 'background-color: #d4edda' if x == "✓ Undervalued" else 'background-color: #f8d7da',
                    subset=['Status']
                ),
                use_container_width=True
            )
            
            # Visualization
            st.markdown("#### Margin of Safety by Model")
            
            # Color by undervalued/overvalued
            df_margin["Color"] = df_margin["Status"].apply(
                lambda x: "Undervalued" if "✓" in x else "Overvalued"
            )
            
            fig2 = px.bar(
                df_margin,
                x="Ticker",
                y="Margin of Safety (%)",
                color="Color",
                text="Margin of Safety (%)",
                facet_col="Model",
                facet_col_wrap=3,
                color_discrete_map={"Undervalued": "#28a745", "Overvalued": "#dc3545"},
                title="Margin of Safety Analysis",
                labels={"Margin of Safety (%)": "Margin of Safety (%)"}
            )
            
            # Add target margin line
            fig2.add_hline(
                y=margin_of_safety_pct * 100,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Target: {margin_of_safety_pct*100:.0f}%"
            )
            
            fig2.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            fig2.update_layout(height=400, showlegend=True)
            fig2.for_each_xaxis(lambda axis: axis.update(tickangle=45))
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # ---- Sensitivity Analysis ----
        if enable_sensitivity and len(tickers) == 1:
            st.markdown("---")
            st.markdown("### 🔍 Sensitivity Analysis")
            
            ticker = tickers[0]
            try:
                df_data = fetch_stock_data(ticker)
                df_iv = IVSimplifier(df_data).simplify()
                vc = ValuationCalculator(df_iv)
                
                # Map parameter names
                param_map = {
                    "Discount Rate": "discount_rate",
                    "Terminal Growth": "terminal_growth",
                    "Growth Rate": "growth_rate"
                }
                
                model_map = {
                    "DCF": "dcf",
                    "DDM Multi-Stage": "ddm_multi_stage"
                }
                
                sensitivity_data = vc.sensitivity_analysis(
                    model=model_map[sensitivity_model],
                    param=param_map[sensitivity_param],
                    range_pct=0.3,
                    steps=7
                )
                
                fig_sensitivity = create_sensitivity_plot(
                    sensitivity_data,
                    f"{ticker} - {sensitivity_model}"
                )
                
                # Add current valuation point
                if sensitivity_model == "DCF":
                    current_val = vc.calculate_dcf(discount_rate=discount_rate, terminal_growth_rate=terminal_growth)
                else:
                    current_val = vc.calculate_ddm()
                
                if current_val:
                    base_param = discount_rate if sensitivity_param == "Discount Rate" else terminal_growth
                    fig_sensitivity.add_trace(go.Scatter(
                        x=[base_param],
                        y=[current_val],
                        mode='markers',
                        name='Current',
                        marker=dict(size=15, color='red', symbol='star')
                    ))
                
                st.plotly_chart(fig_sensitivity, use_container_width=True)
                
                st.info(
                    f"💡 This shows how {ticker}'s intrinsic value changes as "
                    f"the {sensitivity_param.lower()} varies ±30% from the base value."
                )
                
            except Exception as e:
                st.error(f"❌ Sensitivity analysis failed: {str(e)}")
        
        elif enable_sensitivity and len(tickers) > 1:
            st.warning("⚠️ Sensitivity analysis is only available for single ticker analysis")
        
        # ---- Export Options ----
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export intrinsic values to CSV
            csv_iv = df_iv_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Valuations (CSV)",
                data=csv_iv,
                file_name=f"intrinsic_values_{'-'.join(tickers)}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export margin of safety to CSV
            if margin_results:
                csv_margin = df_margin.to_csv(index=False)
                st.download_button(
                    label="📥 Download Margins (CSV)",
                    data=csv_margin,
                    file_name=f"margin_of_safety_{'-'.join(tickers)}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Export complete report
            report_text = f"""
INTRINSIC VALUE ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Tickers: {', '.join(tickers)}

PARAMETERS:
- Discount Rate: {discount_rate*100:.1f}%
- Terminal Growth: {terminal_growth*100:.1f}%
- Target Margin of Safety: {margin_of_safety_pct*100:.1f}%

INTRINSIC VALUES:
{df_iv_results.to_string()}

MARGIN OF SAFETY:
{df_margin.to_string() if margin_results else 'N/A'}
"""
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report_text,
                file_name=f"valuation_report_{'-'.join(tickers)}.txt",
                mime="text/plain"
            )
    
    else:
        st.error("❌ No stocks were successfully analyzed")
    
    # Show failed tickers summary
    if failed_tickers:
        st.markdown("---")
        st.error(f"⚠️ Failed to analyze: {', '.join(failed_tickers)}")
        st.info(
            "💡 **Troubleshooting Tips:**\n"
            "- Verify ticker symbols are correct\n"
            "- Check if the company has sufficient financial history\n"
            "- Some stocks (REITs, financials) may not have all required data\n"
            "- Try again later if there are API connection issues"
        )

# ---- Information Section ----
st.markdown("---")

with st.expander("ℹ️ About the Valuation Models"):
    st.markdown("""
    ### Valuation Models Explained
    
    **1. Discounted Cash Flow (DCF)**
    - Projects future free cash flows and discounts them to present value
    - Best for: Companies with stable, positive cash flows
    - Limitations: Sensitive to growth rate assumptions
    
    **2. Dividend Discount Model (DDM)**
    - Values stock based on present value of future dividends
    - Best for: Dividend-paying companies with consistent payouts
    - Limitations: Not applicable to non-dividend stocks
    
    **3. P/E Multiplier Model**
    - Uses average historical P/E ratio × current EPS
    - Best for: Stable, profitable companies
    - Limitations: Unreliable with volatile earnings
    
    **4. Asset-Based Valuation**
    - Book value (Total Assets - Total Liabilities) per share
    - Best for: Asset-heavy companies, value investing
    - Limitations: Ignores intangible value and future earnings
    
    **5. Modern Graham Formula**
    - Benjamin Graham's formula adjusted for bond yields
    - Best for: Conservative valuation estimate
    - Limitations: May undervalue high-growth companies
    
    ### Confidence Scores
    - **High**: Model has reliable inputs and low volatility
    - **Medium**: Model is usable but has some concerns
    - **Low**: Model results should be interpreted cautiously
    
    ### Margin of Safety
    The margin of safety represents the discount from intrinsic value. 
    A 25% margin means you'd want to buy at 75% of the calculated intrinsic value.
    """)

with st.expander("❓ FAQ"):
    st.markdown("""
    **Q: Why do different models give different values?**  
    A: Each model uses different assumptions and inputs. This is normal and expected. 
    Look at the range and average for a balanced view.
    
    **Q: What if some models show "N/A"?**  
    A: Some models require specific data (e.g., DDM needs dividends). If data is missing, 
    that model cannot be calculated.
    
    **Q: How should I use these valuations?**  
    A: These are estimates, not guarantees. Use them as one input in your investment 
    research, along with qualitative analysis, industry trends, and risk assessment.
    
    **Q: Why does the app warn about data quality?**  
    A: Warnings indicate potential issues with the underlying data (e.g., limited history, 
    negative cash flows). Consider these when interpreting results.
    
    **Q: Can I use this for any stock?**  
    A: These models work best for established companies with financial history. 
    They may not be suitable for:
    - Recent IPOs (< 3 years)
    - REITs (different valuation approach needed)
    - Financial companies (banks, insurance)
    - High-growth startups (no profits yet)
    """)

with st.expander("⚙️ Technical Details"):
    st.markdown(f"""
    ### Default Parameters
    - **Discount Rate**: {DEFAULT_DISCOUNT_RATE*100:.1f}% (Can be adjusted in sidebar)
    - **Terminal Growth Rate**: {DEFAULT_TERMINAL_GROWTH*100:.1f}%
    - **Margin of Safety**: {DEFAULT_MARGIN_OF_SAFETY*100:.0f}%
    - **Data Source**: Yahoo Finance via yfinance library
    - **Cache Duration**: 1 hour (data refreshes after this period)
    
    ### Data Requirements
    - Minimum {MIN_HISTORICAL_YEARS} years of financial data recommended
    - Required metrics: Free Cash Flow, EPS, Share Price, Balance Sheet items
    
    ### Calculation Notes
    - Growth rates calculated using CAGR (Compound Annual Growth Rate)
    - P/E ratios use median to reduce impact of outliers
    - Weighted average gives more weight to high-confidence models
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "⚠️ <b>Disclaimer</b>: This tool is for educational purposes only. "
    "Not financial advice. Always do your own research and consult a financial advisor."
    "</p>",
    unsafe_allow_html=True
)