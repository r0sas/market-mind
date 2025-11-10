# streamlit_intrinsic_value_v4.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import time
import logging

from core.oldscripts.DataFetcher import DataFetcher, DataFetcherError
from core.IV_Simplifier.IV_Simplifier import IVSimplifier, SimplifierError
from core.oldscripts.ValuationCalculator import ValuationCalculator, ValuationError
from core.oldscripts.model_selector import ModelSelector
from core.Config import (
    MODEL_DISPLAY_NAMES,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_MARGIN_OF_SAFETY,
    MIN_HISTORICAL_YEARS
)

# Configure logging
logger = logging.getLogger(__name__)

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

# Model Selection Mode
st.sidebar.subheader("🧭 Model Selection Mode")

use_smart_selection = st.sidebar.radio(
    "How should models be selected?",
    options=["🤖 Auto-select (Recommended)", "✋ Manual selection"],
    help="Auto-select uses AI to choose the best models for each company"
) == "🤖 Auto-select (Recommended)"

if use_smart_selection:
    st.sidebar.success("✨ Smart selection enabled")
    
    min_fit_score = st.sidebar.slider(
        "Minimum Model Fit Score",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Only show models scoring above this threshold (0.5 = recommended)"
    )
    
    show_all_scores = st.sidebar.checkbox(
        "Show excluded models",
        value=False,
        help="Display fit scores for models that didn't meet the threshold"
    )
    
    selected_models = None  # Will be determined per company
    
else:
    st.sidebar.info("Manual selection active")
    
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
    model_selection_info = {}  # NEW: Store model selection data
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
            
            # NEW: Model Selection Logic
            if use_smart_selection:
                selector = ModelSelector(df_iv)
                fit_scores = selector.calculate_fit_scores()
                recommended = selector.get_recommended_models(min_score=min_fit_score)
                explanations = selector.get_fit_explanations()
                exclusions = selector.get_exclusion_reasons()
                
                # Store for visualization
                model_selection_info[ticker] = {
                    'fit_scores': fit_scores,
                    'recommended': recommended,
                    'explanations': explanations,
                    'exclusions': exclusions,
                    'selector': selector  # Store for later use
                }
                
                models_to_calculate = recommended
                
                if not models_to_calculate:
                    st.warning(
                        f"⚠️ {ticker}: No models meet minimum fit score ({min_fit_score:.1f}). "
                        f"Try lowering the threshold or check data quality."
                    )
                    continue
            else:
                # Manual selection
                model_name_map = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
                models_to_calculate = [model_name_map[m] for m in selected_models if m in model_name_map]
            
            # Calculate valuations
            vc = ValuationCalculator(df_iv)
            vc.calculate_all_valuations(
                models_to_calculate=models_to_calculate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth
            )
            
            avg_value = vc.get_average_valuation(weighted=use_weighted_avg)
            current_price = vc.current_price
            iv_values = vc.get_results()
            confidence_scores = vc.get_confidence_scores()
            model_warnings = vc.get_model_warnings()
            margin_analysis = vc.get_margin_of_safety(target_margin=margin_of_safety_pct)
            
            # Prepare intrinsic value results
            iv_filtered = {
                "Ticker": ticker,
                "Current Price": current_price,
                "Years of Data": quality_report.get('num_years', 'N/A')
            }
            
            # Add model selection info if smart selection is enabled
            if use_smart_selection:
                iv_filtered["Models Used"] = len(models_to_calculate)
            
            # Reverse mapping for display names
            model_name_map = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
            
            # Add valuation results to display
            for model_key, model_value in iv_values.items():
                display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
                
                # SAFETY CHECK: Skip negative or zero valuations (invalid results)
                if model_value is None or model_value <= 0:
                    logger.warning(f"Skipping {model_key} for {ticker}: invalid value ({model_value})")
                    continue
                
                # In smart selection, show all calculated models
                # In manual selection, only show selected models
                if use_smart_selection or display_name in selected_models:
                    iv_filtered[display_name] = model_value
                    
                    if show_confidence and model_key in confidence_scores:
                        iv_filtered[f"{display_name}_Confidence"] = confidence_scores[model_key]
            
            # Add average
            if avg_value:
                avg_label = "Weighted Average" if use_weighted_avg else "Average"
                iv_filtered[avg_label] = avg_value
            
            results.append(iv_filtered)
            
            # Prepare margin of safety results
            if margin_analysis:
                for model, data in margin_analysis.items():
                    display_name = MODEL_DISPLAY_NAMES.get(model, model)
                    
                    # SAFETY CHECK: Skip if intrinsic value is invalid
                    if data["intrinsic_value"] is None or data["intrinsic_value"] <= 0:
                        logger.warning(f"Skipping {model} margin analysis for {ticker}: invalid intrinsic value")
                        continue
                    
                    # In smart selection, show all calculated models
                    # In manual selection, only show selected models
                    if use_smart_selection or display_name in selected_models:
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
        format_dict = {col: "${:,.2f}" for col in df_display.columns if col not in ["Years of Data", "Models Used"]}
        
        st.dataframe(
            df_display.style.format(format_dict).background_gradient(
                subset=[col for col in df_display.columns if col not in ["Years of Data", "Models Used"]],
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
        plot_cols = [col for col in df_display.columns if col not in ["Years of Data", "Current Price", "Models Used"]]
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
        
        # ---- Model Selection Analysis (NEW SECTION) ----
        if use_smart_selection and model_selection_info:
            st.markdown("### 🧭 Model Selection Analysis")
            st.markdown("*Understanding why specific models were chosen for each company*")
            
            for ticker, info in model_selection_info.items():
                with st.expander(f"📊 {ticker} - Model Selection Details", expanded=False):
                    
                    # Create fit score dataframe
                    fit_data = []
                    for model, score in info['fit_scores'].items():
                        model_display = MODEL_DISPLAY_NAMES.get(model, model)
                        is_recommended = model in info['recommended']
                        
                        # Determine status
                        if score >= 0.7:
                            status_emoji = "🟢"
                            status_text = "Highly Recommended"
                        elif score >= 0.5:
                            status_emoji = "🟡"
                            status_text = "Recommended"
                        elif score >= 0.3:
                            status_emoji = "🟠"
                            status_text = "Marginal"
                        else:
                            status_emoji = "🔴"
                            status_text = "Not Suitable"
                        
                        fit_data.append({
                            'Model': model_display,
                            'Fit Score': score,
                            'Status': f"{status_emoji} {status_text}",
                            'Selected': '✅' if is_recommended else '❌',
                            'Score_Numeric': score
                        })
                    
                    fit_df = pd.DataFrame(fit_data).sort_values('Score_Numeric', ascending=False)
                    
                    # Layout: Chart on left, metrics on right
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Horizontal bar chart of fit scores
                        fig_fit = go.Figure()
                        
                        # Add bars with color based on score
                        colors = []
                        for score in fit_df['Score_Numeric']:
                            if score >= 0.7:
                                colors.append('#28a745')  # Green
                            elif score >= 0.5:
                                colors.append('#ffc107')  # Yellow
                            elif score >= 0.3:
                                colors.append('#fd7e14')  # Orange
                            else:
                                colors.append('#dc3545')  # Red
                        
                        fig_fit.add_trace(go.Bar(
                            y=fit_df['Model'],
                            x=fit_df['Fit Score'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=fit_df['Fit Score'].apply(lambda x: f'{x:.2f}'),
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Fit Score: %{x:.2f}<extra></extra>'
                        ))
                        
                        # Add threshold line
                        fig_fit.add_vline(
                            x=min_fit_score,
                            line_dash="dash",
                            line_color="orange",
                            line_width=2,
                            annotation_text=f"Threshold ({min_fit_score:.1f})",
                            annotation_position="top right"
                        )
                        
                        fig_fit.update_layout(
                            title=f"{ticker} - Model Fit Scores",
                            xaxis_title="Fit Score (0.0 - 1.0)",
                            yaxis_title="",
                            height=350,
                            showlegend=False,
                            xaxis=dict(range=[0, 1.1])
                        )
                        
                        st.plotly_chart(fig_fit, use_container_width=True)
                    
                    with col2:
                        # Summary metrics
                        st.metric(
                            "Models Selected",
                            len(info['recommended']),
                            f"of {len(info['fit_scores'])}"
                        )
                        
                        st.markdown("**Score Legend:**")
                        st.markdown("🟢 **0.70+** Highly Recommended")
                        st.markdown("🟡 **0.50-0.69** Recommended")
                        st.markdown("🟠 **0.30-0.49** Marginal")
                        st.markdown("🔴 **<0.30** Not Suitable")
                        
                        st.markdown("---")
                        
                        # Quick stats
                        avg_score = sum(info['fit_scores'].values()) / len(info['fit_scores'])
                        st.metric("Average Fit Score", f"{avg_score:.2f}")
                    
                    # Detailed explanations
                    st.markdown("---")
                    st.markdown("#### 📋 Detailed Analysis")
                    
                    # Show recommended models with explanations
                    if info['recommended']:
                        st.markdown("**✅ Selected Models:**")
                        
                        for model in info['recommended']:
                            model_display = MODEL_DISPLAY_NAMES.get(model, model)
                            score = info['fit_scores'][model]
                            
                            with st.container():
                                # Model header with score badge
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.markdown(f"**{model_display}**")
                                with col_b:
                                    if score >= 0.7:
                                        st.success(f"Score: {score:.2f}")
                                    else:
                                        st.warning(f"Score: {score:.2f}")
                                
                                # Explanations
                                if model in info['explanations']:
                                    exp = info['explanations'][model]
                                    
                                    if exp['pass']:
                                        st.markdown("*Strengths:*")
                                        for reason in exp['pass']:
                                            st.markdown(f"✓ {reason}")
                                    
                                    if exp['fail']:
                                        st.markdown("*Considerations:*")
                                        for reason in exp['fail']:
                                            st.markdown(f"⚠ {reason}")
                                
                                st.markdown("")  # Spacing
                    
                    # Show excluded models if requested
                    if show_all_scores and info['exclusions']:
                        st.markdown("---")
                        st.markdown("**❌ Excluded Models:**")
                        
                        for model, reason in info['exclusions'].items():
                            model_display = MODEL_DISPLAY_NAMES.get(model, model)
                            score = info['fit_scores'].get(model, 0.0)
                            
                            with st.expander(f"{model_display} (Fit: {score:.2f})"):
                                st.error(f"**Primary Reason:** {reason}")
                                
                                if model in info['explanations']:
                                    exp = info['explanations'][model]
                                    
                                    if exp['fail']:
                                        st.markdown("**Issues Identified:**")
                                        for detail in exp['fail']:
                                            st.markdown(f"• {detail}")
                                    
                                    if exp['pass']:
                                        st.markdown("**Positive Factors:**")
                                        for detail in exp['pass']:
                                            st.markdown(f"• {detail}")
        
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
- Selection Mode: {'Auto-select' if use_smart_selection else 'Manual'}

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
    - **Auto-Selected When:** Company has positive FCF in 60%+ of years
    
    **2. Dividend Discount Model (DDM)**
    - Values stock based on present value of future dividends
    - **Single-Stage:** For stable dividend payers (growth <8%)
    - **Multi-Stage:** For growing dividend payers (growth ≥8%)
    - Best for: Dividend-paying companies with consistent payouts
    - Limitations: Not applicable to non-dividend stocks
    - **Auto-Selected When:** Company pays dividends consistently
    
    **3. P/E Multiplier Model**
    - Uses average historical P/E ratio × current EPS
    - Best for: Stable, profitable companies
    - Limitations: Unreliable with volatile earnings
    - **Auto-Selected When:** Positive earnings in 60%+ of years, reasonable P/E range
    
    **4. Asset-Based Valuation**
    - Book value (Total Assets - Total Liabilities) per share
    - Best for: Asset-heavy companies, value investing, distressed situations
    - Limitations: Ignores intangible value and future earnings
    - **Auto-Selected When:** Low profitability but strong asset base
    
    **5. Modern Graham Formula**
    - Benjamin Graham's formula adjusted for bond yields
    - Best for: Conservative valuation estimate
    - Limitations: May undervalue high-growth companies
    - **Auto-Selected When:** 5+ years of profitability, strong ROE, low debt
    
    ### Confidence Scores (Model Quality)
    - **High**: Model has reliable inputs and low volatility
    - **Medium**: Model is usable but has some concerns
    - **Low**: Model results should be interpreted cautiously
    
    ### Fit Scores (Model Appropriateness) 🆕
    Smart selection uses fit scores to determine which models are appropriate:
    - **0.70-1.00**: Highly recommended - All criteria met
    - **0.50-0.69**: Recommended - Most criteria met
    - **0.30-0.49**: Marginal - Some concerns exist
    - **0.00-0.29**: Not suitable - Critical criteria missing
    
    The fit score considers:
    - **Primary Criteria (50%)**: Core requirements (e.g., has dividends for DDM)
    - **Supporting Criteria (30%)**: Quality indicators (growth, stability)
    - **Data Quality (20%)**: Years of history, completeness
    
    ### Margin of Safety
    The margin of safety represents the discount from intrinsic value. 
    A 25% margin means you'd want to buy at 75% of the calculated intrinsic value.
    """)

with st.expander("❓ FAQ"):
    st.markdown("""
    **Q: What's the difference between Confidence Scores and Fit Scores?**  
    A: 
    - **Fit Score**: Measures if the model is *appropriate* for this company (e.g., does it pay dividends for DDM?)
    - **Confidence Score**: Measures how *reliable* the model's output is (e.g., is the data volatile?)
    
    **Q: Why do different models give different values?**  
    A: Each model uses different assumptions and inputs. This is normal and expected. 
    Look at the range and average for a balanced view.
    
    **Q: What if some models show "N/A" or are excluded?**  
    A: Some models require specific data (e.g., DDM needs dividends). If data is missing 
    or the company doesn't fit the model's criteria, it won't be calculated.
    
    **Q: Should I use Auto-select or Manual selection?**  
    A: 
    - **Auto-select (Recommended)**: Let the system choose based on financial analysis. Best for most users.
    - **Manual**: Override if you have specific models you want to see, or for educational comparison.
    
    **Q: Why are fewer models selected with Smart Selection?**  
    A: That's the point! Smart selection only shows models that are truly appropriate for the company.
    Showing fewer, more relevant models is better than showing all models with some being meaningless.
    
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
    
    **Q: What if NO models are selected for my stock?**  
    A: This means the company doesn't meet minimum criteria for any model. This could indicate:
    - Very limited financial history
    - Highly volatile or inconsistent financials
    - Missing critical data
    - Company may not be suitable for traditional valuation methods
    Try lowering the fit score threshold or check data quality warnings.
    """)

with st.expander("⚙️ Technical Details"):
    st.markdown(f"""
    ### Default Parameters
    - **Discount Rate**: {DEFAULT_DISCOUNT_RATE*100:.1f}% (Can be adjusted in sidebar)
    - **Terminal Growth Rate**: {DEFAULT_TERMINAL_GROWTH*100:.1f}%
    - **Margin of Safety**: {DEFAULT_MARGIN_OF_SAFETY*100:.0f}%
    - **Data Source**: Yahoo Finance via yfinance library
    - **Cache Duration**: 1 hour (data refreshes after this period)
    
    ### Smart Model Selection Thresholds
    **DCF Model:**
    - Minimum 3 positive FCF years (60% of total)
    - Positive revenue growth
    - Payout ratio < 80%
    
    **DDM Single-Stage:**
    - 5+ years of dividend history
    - Dividend growth < 8% (stable)
    - Payout ratio 30-70%
    
    **DDM Multi-Stage:**
    - 3+ years of dividend history
    - Dividend growth ≥ 8%
    - Positive EPS growth
    
    **P/E Model:**
    - 3+ years of positive earnings (60% of total)
    - P/E ratio between 5-50
    - P/E volatility < 1.5 (coefficient of variation)
    
    **Graham Formula:**
    - 5+ years of consistent profitability
    - ROE > 10%
    - Debt-to-Equity < 0.5
    
    **Asset-Based:**
    - Positive tangible book value
    - Asset quality > 30% ((Assets-Liabilities)/Assets)
    - Most relevant when net margin < 5%
    
    ### Data Requirements
    - Minimum {MIN_HISTORICAL_YEARS} years of financial data recommended
    - Required metrics: Free Cash Flow, EPS, Share Price, Balance Sheet items
    
    ### Calculation Notes
    - Growth rates calculated using CAGR (Compound Annual Growth Rate)
    - P/E ratios use median to reduce impact of outliers
    - Weighted average gives more weight to high-confidence models
    - Fit scores combine multiple criteria with 50% primary, 30% supporting, 20% data quality weights
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