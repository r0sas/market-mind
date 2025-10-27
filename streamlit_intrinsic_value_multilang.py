# streamlit_intrinsic_value_v5_multilang.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import logging

from core.DataFetcher import DataFetcher, DataFetcherError
from core.IVSimplifier import IVSimplifier, SimplifierError
from core.ValuationCalculator import ValuationCalculator, ValuationError
from core.model_selector import ModelSelector
from core.ai_insights import AIInsightsGenerator
from core.ai_parameter_optimizer import AIParameterOptimizer
from core.ai_visual_explainer import AIVisualExplainer
from core.Config import (
    MODEL_DISPLAY_NAMES,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_TERMINAL_GROWTH,
    DEFAULT_MARGIN_OF_SAFETY,
    MIN_HISTORICAL_YEARS,
    ENABLE_AI_INSIGHTS
)
from translations import get_text, get_available_languages

# Configure logging
logger = logging.getLogger(__name__)

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Language selector in the top-right corner
col_lang1, col_lang2 = st.columns([5, 1])
with col_lang2:
    languages = get_available_languages()
    selected_lang = st.selectbox(
        '🌐',
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(st.session_state.language),
        key='lang_selector'
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

# Get translation function for current language
def t(key: str, **kwargs) -> str:
    """Shorthand for get_text with current language."""
    return get_text(key, st.session_state.language, **kwargs)

# Page configuration
st.set_page_config(
    page_title=t('page_title'),
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
        title=f"{t('sensitivity_analysis')}: {model_name}",
        xaxis_title=sensitivity_data['parameter'].replace('_', ' ').title(),
        yaxis_title=t('iv_comparison') + " ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


# ---- Sidebar Configuration ----
st.sidebar.title(t('sidebar_title'))
st.sidebar.markdown("---")

# Model Selection Mode
st.sidebar.subheader(t('model_selection_mode'))

selection_mode = st.sidebar.radio(
    t('model_selection_question'),
    options=[t('auto_select'), t('manual_select'), "🤖 AI Mode (Smart Parameters)"],
    help=t('auto_select_help')
)

use_smart_selection = selection_mode == t('auto_select')
use_ai_mode = selection_mode == "🤖 AI Mode (Smart Parameters)"

if use_smart_selection:
    st.sidebar.success(t('smart_enabled'))
    
    min_fit_score = st.sidebar.slider(
        t('min_fit_score'),
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help=t('min_fit_score_help')
    )
    
    show_all_scores = st.sidebar.checkbox(
        t('show_excluded'),
        value=False,
        help=t('show_excluded_help')
    )
    
    selected_models = None
    
elif use_ai_mode:
    st.sidebar.success("✨ AI Mode: Smart model selection + optimized parameters")
    
    min_fit_score = st.sidebar.slider(
        t('min_fit_score'),
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help=t('min_fit_score_help')
    )
    
    show_all_scores = st.sidebar.checkbox(
        t('show_excluded'),
        value=False,
        help=t('show_excluded_help')
    )
    
    st.sidebar.info("💡 AI will optimize discount rate, terminal growth, and projection years for each company")
    
    selected_models = None
    
else:
    st.sidebar.info(t('manual_active'))
    
    st.sidebar.subheader(t('model_selection'))
    all_models = list(MODEL_DISPLAY_NAMES.values())
    selected_models = st.sidebar.multiselect(
        t('select_models'),
        all_models,
        default=all_models,
        help=t('select_models_help')
    )

st.sidebar.markdown("---")
st.sidebar.subheader(t('advanced_params'))

with st.sidebar.expander(t('dcf_params')):
    discount_rate = st.slider(
        t('discount_rate'),
        min_value=5.0,
        max_value=20.0,
        value=DEFAULT_DISCOUNT_RATE * 100,
        step=0.5,
        help=t('discount_rate_help')
    ) / 100
    
    terminal_growth = st.slider(
        t('terminal_growth'),
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_TERMINAL_GROWTH * 100,
        step=0.25,
        help=t('terminal_growth_help')
    ) / 100

with st.sidebar.expander(t('analysis_options')):
    show_confidence = st.checkbox(t('show_confidence'), value=True)
    show_warnings = st.checkbox(t('show_warnings'), value=True)
    use_weighted_avg = st.checkbox(
        t('weighted_avg'),
        value=False,
        help=t('weighted_avg_help')
    )
    
    margin_of_safety_pct = st.slider(
        t('margin_safety'),
        min_value=10,
        max_value=50,
        value=int(DEFAULT_MARGIN_OF_SAFETY * 100),
        step=5,
        help=t('margin_safety_help')
    ) / 100

st.sidebar.markdown("---")
st.sidebar.subheader(t('ai_insights'))

enable_ai_insights = st.sidebar.checkbox(
    t('enable_ai'),
    value=ENABLE_AI_INSIGHTS,
    help=t('enable_ai_help')
)

if enable_ai_insights:
    groq_api_key = st.sidebar.text_input(
        t('groq_key'),
        type="password",
        help=t('groq_key_help'),
        placeholder="gsk_..."
    )
    
    if groq_api_key:
        st.sidebar.success(t('api_provided'))
    else:
        st.sidebar.info(t('api_info'))
else:
    groq_api_key = None

with st.sidebar.expander(t('sensitivity')):
    enable_sensitivity = st.checkbox(t('enable_sensitivity'), value=False)
    if enable_sensitivity:
        sensitivity_model = st.selectbox(
            t('model_analyze'),
            ["DCF", "DDM Multi-Stage"]
        )
        sensitivity_param = st.selectbox(
            t('param_vary'),
            ["Discount Rate", "Terminal Growth", "Growth Rate"]
        )

st.sidebar.markdown("---")
st.sidebar.markdown(t('tip'), unsafe_allow_html=True)

# ---- Main Page ----
st.title(f"📈 {t('page_title')}")
st.markdown(t('page_subtitle'))

# Ticker input
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        t('enter_tickers'),
        value="AAPL",
        placeholder=t('ticker_placeholder'),
        help=t('ticker_help')
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    calculate_button = st.button(
        t('calculate'), 
        type="primary", 
        use_container_width=True
    )

if calculate_button:
    # Validate input
    if not ticker_input.strip():
        st.error(t('enter_ticker_error'))
        st.stop()
    
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",")]
    tickers = list(dict.fromkeys(tickers))  # Remove duplicates
    
    st.markdown("---")
    
    # Initialize results storage
    results = []
    margin_results = []
    all_warnings = {}
    model_selection_info = {}
    ai_insights = {}
    ai_parameters = {}
    failed_tickers = []
    
    # Initialize AI generator if enabled
    ai_generator = None
    if enable_ai_insights and groq_api_key:
        try:
            ai_generator = AIInsightsGenerator(api_key=groq_api_key)
            if ai_generator.test_connection():
                st.info(t('ai_enabled'))
            else:
                st.warning(t('ai_failed'))
                ai_generator = None
        except Exception as e:
            st.error(f"{t('ai_init_failed')}: {e}")
            ai_generator = None
    
    # Initialize AI Parameter Optimizer and Visual Explainer if in AI Mode
    param_optimizer = None
    visual_explainer = None
    
    if use_ai_mode and groq_api_key:
        try:
            param_optimizer = AIParameterOptimizer(api_key=groq_api_key, use_ai=True)
            visual_explainer = AIVisualExplainer(api_key=groq_api_key)
            st.info("🤖 AI Parameter Optimization enabled")
        except Exception as e:
            st.warning(f"⚠️ AI Parameter Optimization unavailable: {e}")
            st.info("Using rule-based parameter optimization instead")
            param_optimizer = AIParameterOptimizer(use_ai=False)
            visual_explainer = AIVisualExplainer()
    elif use_ai_mode:
        param_optimizer = AIParameterOptimizer(use_ai=False)
        visual_explainer = AIVisualExplainer()
        st.info("Using rule-based parameter optimization (no API key)")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"{t('processing')} {ticker}... ({idx + 1}/{len(tickers)})")
        
        try:
            # Fetch data
            with st.spinner(f"{t('fetching')} {ticker}..."):
                df_data = fetch_stock_data(ticker)
            
            # Simplify data
            simplifier = IVSimplifier(df_data)
            df_iv = simplifier.simplify()
            
            # Get data quality info
            quality_report = simplifier.get_data_quality_report()
            all_warnings[ticker] = quality_report.get('warnings', [])
            
            # AI Mode - Optimize Parameters
            company_discount_rate = discount_rate
            company_terminal_growth = terminal_growth
            
            if use_ai_mode and param_optimizer:
                try:
                    temp_fetcher = DataFetcher(ticker)
                    company_info = temp_fetcher.get_info()
                    
                    financial_metrics = {
                        'beta': company_info.get('beta', 1.0),
                        'debt_to_equity': company_info.get('debtToEquity', 0) / 100 if company_info.get('debtToEquity') else 0,
                        'revenue_cagr': 0.10,
                        'fcf_cagr': 0.08,
                        'roe': company_info.get('returnOnEquity', 0),
                        'net_margin': company_info.get('profitMargins', 0),
                        'total_debt': company_info.get('totalDebt', 0),
                        'interest_expense': 0,
                        'tax_rate': 0.21
                    }
                    
                    company_data = {
                        'sector': company_info.get('sector', 'default'),
                        'industry': company_info.get('industry', 'Unknown'),
                        'marketCap': company_info.get('marketCap', 0)
                    }
                    
                    optimized = param_optimizer.optimize_parameters(
                        ticker, company_data, financial_metrics
                    )
                    
                    ai_parameters[ticker] = optimized
                    company_discount_rate = optimized['discount_rate']
                    company_terminal_growth = optimized['terminal_growth']
                    
                    st.info(f"🤖 {ticker}: Discount Rate = {company_discount_rate:.2%}, Terminal Growth = {company_terminal_growth:.2%} ({optimized['method']})")
                    
                except Exception as e:
                    logger.error(f"AI parameter optimization failed for {ticker}: {e}")
                    st.warning(f"⚠️ Using default parameters for {ticker}")
            
            # Model Selection Logic
            if use_smart_selection or use_ai_mode:
                selector = ModelSelector(df_iv)
                fit_scores = selector.calculate_fit_scores()
                recommended = selector.get_recommended_models(min_score=min_fit_score)
                explanations = selector.get_fit_explanations()
                exclusions = selector.get_exclusion_reasons()
                
                model_selection_info[ticker] = {
                    'fit_scores': fit_scores,
                    'recommended': recommended,
                    'explanations': explanations,
                    'exclusions': exclusions,
                    'selector': selector
                }
                
                models_to_calculate = recommended
                
                if not models_to_calculate:
                    st.warning(
                        f"⚠️ {ticker}: No models meet minimum fit score ({min_fit_score:.1f}). "
                        f"Try lowering the threshold or check data quality."
                    )
                    continue
            else:
                model_name_map = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
                models_to_calculate = [model_name_map[m] for m in selected_models if m in model_name_map]
            
            # Calculate valuations with company-specific parameters
            vc = ValuationCalculator(df_iv)
            vc.calculate_all_valuations(
                models_to_calculate=models_to_calculate,
                discount_rate=company_discount_rate,
                terminal_growth_rate=company_terminal_growth
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
                t('current_price'): current_price,
                "Years of Data": quality_report.get('num_years', 'N/A')
            }
            
            if use_smart_selection or use_ai_mode:
                iv_filtered[t('models_selected')] = len(models_to_calculate)
            
            for model_key, model_value in iv_values.items():
                display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
                
                if model_value is None or model_value <= 0:
                    logger.warning(f"Skipping {model_key} for {ticker}: invalid value")
                    continue
                
                if use_smart_selection or use_ai_mode or display_name in selected_models:
                    iv_filtered[display_name] = model_value
                    
                    if show_confidence and model_key in confidence_scores:
                        iv_filtered[f"{display_name}_Confidence"] = confidence_scores[model_key]
            
            # Add average
            if avg_value:
                avg_label = "Weighted Average" if use_weighted_avg else "Average"
                iv_filtered[avg_label] = avg_value
            
            results.append(iv_filtered)
            
            # Generate AI insights
            if ai_generator and iv_values:
                try:
                    insight_fetcher = DataFetcher(ticker)
                    company_info = insight_fetcher.get_info()
                    sector = company_info.get('sector', 'Unknown')
                    
                    insight = ai_generator.generate_insights(
                        ticker=ticker,
                        current_price=current_price,
                        valuations=iv_values,
                        sector=sector,
                        warnings=quality_report.get('warnings', []),
                        confidence_scores=confidence_scores
                    )
                    
                    if insight:
                        ai_insights[ticker] = insight
                        logger.info(f"Generated AI insights for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to generate AI insights for {ticker}: {e}")
            
            # Prepare margin of safety results
            if margin_analysis:
                for model, data in margin_analysis.items():
                    display_name = MODEL_DISPLAY_NAMES.get(model, model)
                    
                    if data["intrinsic_value"] is None or data["intrinsic_value"] <= 0:
                        continue
                    
                    if use_smart_selection or use_ai_mode or display_name in selected_models:
                        margin_results.append({
                            "Ticker": ticker,
                            "Model": display_name,
                            "Intrinsic Value": data["intrinsic_value"],
                            t('current_price'): data["current_price"],
                            "Margin of Safety (%)": round(data["margin_of_safety"] * 100, 1),
                            "Target Buy Price": data["target_buy_price"],
                            "Status": "✓ Undervalued" if data["is_undervalued"] else "✗ Overvalued",
                            "Confidence": data.get("confidence", "N/A")
                        })
        
        except (DataFetcherError, SimplifierError, ValuationError) as e:
            st.error(f"❌ Error {t('processing')} {ticker}: {str(e)}")
            failed_tickers.append(ticker)
        except Exception as e:
            st.error(f"❌ {ticker}: {str(e)}")
            failed_tickers.append(ticker)
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if results:
        st.success(f"{t('success')} {len(results)} {t('stocks')}")
        
        # Display warnings
        if show_warnings and any(all_warnings.values()):
            st.markdown(f"### {t('data_warnings')}")
            for ticker, warnings in all_warnings.items():
                if warnings:
                    with st.expander(f"{ticker} - {len(warnings)} {t('warnings')}"):
                        for warning in warnings:
                            st.warning(warning)
        
        st.markdown("---")
        
        # AI Parameter Optimization Results
        if use_ai_mode and ai_parameters:
            st.markdown("### 🤖 AI-Optimized Parameters")
            st.markdown("*Company-specific discount rates and terminal growth rates*")
            
            param_data = []
            for ticker, params in ai_parameters.items():
                param_data.append({
                    'Ticker': ticker,
                    'Discount Rate': f"{params['discount_rate']:.2%}",
                    'Terminal Growth': f"{params['terminal_growth']:.2%}",
                    'Projection Years': params['projection_years'],
                    'Method': params['method'],
                    'Confidence': params['confidence']
                })
            
            df_params = pd.DataFrame(param_data)
            
            st.dataframe(
                df_params.style.applymap(
                    lambda x: 'background-color: #d4edda' if x == 'High' else 'background-color: #fff3cd' if x == 'Medium' else '',
                    subset=['Confidence']
                ),
                use_container_width=True,
                hide_index=True
            )
            
            with st.expander("📋 View Parameter Explanations"):
                for ticker, params in ai_parameters.items():
                    st.markdown(f"#### {ticker}")
                    st.markdown(params['explanation'])
                    
                    if 'ai_reasoning' in params and params['ai_reasoning']:
                        with st.expander(f"🧠 AI Reasoning for {ticker}"):
                            st.markdown(params['ai_reasoning'])
                    
                    if 'adjustments' in params:
                        adj = params['adjustments']
                        st.caption(f"Base: {adj['base_rate']:.2%} | Size: +{adj['size_adjustment']:.2%} | Beta: {adj['beta_adjustment']:+.2%} | Leverage: +{adj['leverage_adjustment']:.2%}")
                    
                    st.markdown("---")
            
            # AI Mode - Explain parameters for beginners
            if visual_explainer:
                st.markdown("#### 🎓 Understanding These Parameters (Beginner Guide)")
                
                for ticker, params in ai_parameters.items():
                    explanation = visual_explainer.explain_optimized_parameters(
                        ticker=ticker,
                        discount_rate=params['discount_rate'],
                        terminal_growth=params['terminal_growth'],
                        explanation=params['explanation']
                    )
                    
                    if explanation:
                        with st.expander(f"💡 Why these settings for {ticker}?"):
                            st.info(explanation)
        
        st.markdown("---")
        
        # AI-Powered Insights Section
        if enable_ai_insights and ai_insights:
            st.markdown(f"### {t('ai_analysis')}")
            st.markdown(t('ai_subtitle'))
            
            for ticker, insight in ai_insights.items():
                with st.expander(f"💡 {ticker} - AI Analysis", expanded=True):
                    st.markdown(insight)
                    st.caption(t('ai_caption'))
        
        st.markdown("---")
        
        # Intrinsic Value Table
        st.markdown(f"### {t('iv_summary')}")
        
        df_iv_results = pd.DataFrame(results)
        
        value_cols = [col for col in df_iv_results.columns if not col.endswith('_Confidence')]
        confidence_cols = [col for col in df_iv_results.columns if col.endswith('_Confidence')]
        
        df_display = df_iv_results[value_cols].set_index("Ticker")
        
        format_dict = {col: "${:,.2f}" for col in df_display.columns if col not in ["Years of Data", t('models_selected')]}
        
        st.dataframe(
            df_display.style.format(format_dict).background_gradient(
                subset=[col for col in df_display.columns if col not in ["Years of Data", t('models_selected')]],
                cmap="RdYlGn",
                axis=1
            ),
            use_container_width=True
        )
        
        # AI Mode - Explain the table for beginners
        if use_ai_mode and visual_explainer:
            st.markdown("#### 🎓 Understanding This Table (Beginner Guide)")
            
            for ticker_row in results:
                ticker = ticker_row['Ticker']
                current = ticker_row.get(t('current_price'), 0)
                
                ticker_valuations = {
                    k: v for k, v in ticker_row.items()
                    if k not in ['Ticker', t('current_price'), 'Years of Data', t('models_selected'), 'Average', 'Weighted Average']
                    and not k.endswith('_Confidence')
                }
                
                avg_val = ticker_row.get('Average') or ticker_row.get('Weighted Average', 0)
                
                if ticker_valuations and current and avg_val:
                    explanation = visual_explainer.explain_intrinsic_value_table(
                        ticker=ticker,
                        current_price=current,
                        valuations=ticker_valuations,
                        average_value=avg_val
                    )
                    
                    if explanation:
                        with st.expander(f"💡 What does this mean for {ticker}?", expanded=(len(results)==1)):
                            st.info(explanation)
        
        # Confidence scores table
        if show_confidence and confidence_cols:
            with st.expander(t('confidence_scores')):
                confidence_data = []
                for _, row in df_iv_results.iterrows():
                    conf_row = {"Ticker": row["Ticker"]}
                    for conf_col in confidence_cols:
                        model_name = conf_col.replace("_Confidence", "")
                        conf_row[model_name] = row[conf_col]
                    confidence_data.append(conf_row)
                
                df_confidence = pd.DataFrame(confidence_data).set_index("Ticker")
                
                def color_confidence(val):
                    colors = {
                        'High': 'background-color: #d4edda', 
                        'Medium': 'background-color: #fff3cd',
                        'Low': 'background-color: #f8d7da'
                    }
                    return colors.get(val, '')
                
                st.dataframe(
                    df_confidence.style.applymap(color_confidence),
                    use_container_width=True
                )
        
        # Visualization
        st.markdown(f"### {t('iv_comparison')}")
        
        plot_cols = [col for col in df_display.columns if col not in ["Years of Data", t('current_price'), t('models_selected')]]
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
            title=t('iv_comparison'),
            labels={"Value": "Value ($)", "Model": "Valuation Model"}
        )
        
        if t('current_price') in df_display.columns:
            for ticker in df_display.index:
                current = df_display.loc[ticker, t('current_price')]
                fig.add_hline(
                    y=current,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{ticker} {t('current_price')}: ${current:.2f}",
                    annotation_position="right"
                )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Mode - Explain the comparison chart
        if use_ai_mode and visual_explainer and len(results) > 1:
            st.markdown("#### 🎓 Understanding This Chart (Beginner Guide)")
            
            comparison_valuations = {}
            comparison_prices = {}
            
            for ticker_row in results:
                ticker = ticker_row['Ticker']
                comparison_prices[ticker] = ticker_row.get(t('current_price'), 0)
                
                comparison_valuations[ticker] = {
                    k: v for k, v in ticker_row.items()
                    if k not in ['Ticker', t('current_price'), 'Years of Data', t('models_selected'), 'Average', 'Weighted Average']
                    and not k.endswith('_Confidence')
                }
            
            explanation = visual_explainer.explain_comparison_chart(
                tickers=list(comparison_prices.keys()),
                valuations=comparison_valuations,
                current_prices=comparison_prices
            )
            
            if explanation:
                st.info(explanation)
        
        st.markdown("---")
        
        # Model Selection Analysis
        if (use_smart_selection or use_ai_mode) and model_selection_info:
            st.markdown(f"### {t('model_analysis')}")
            st.markdown(t('model_analysis_subtitle'))
            
            for ticker, info in model_selection_info.items():
                with st.expander(f"📊 {ticker} - {t('model_details')}", expanded=False):
                    
                    fit_data = []
                    for model, score in info['fit_scores'].items():
                        model_display = MODEL_DISPLAY_NAMES.get(model, model)
                        is_recommended = model in info['recommended']
                        
                        if score >= 0.7:
                            status_emoji, status_text = "🟢", t('highly_recommended').replace('**', '').replace('0.70+', '').strip()
                        elif score >= 0.5:
                            status_emoji, status_text = "🟡", t('recommended').replace('**', '').replace('0.50-0.69', '').strip()
                        elif score >= 0.3:
                            status_emoji, status_text = "🟠", t('marginal').replace('**', '').replace('0.30-0.49', '').strip()
                        else:
                            status_emoji, status_text = "🔴", t('not_suitable').replace('**', '').replace('<0.30', '').strip()
                        
                        fit_data.append({
                            'Model': model_display,
                            'Fit Score': score,
                            'Status': f"{status_emoji} {status_text}",
                            'Selected': '✅' if is_recommended else '❌',
                            'Score_Numeric': score
                        })
                    
                    fit_df = pd.DataFrame(fit_data).sort_values('Score_Numeric', ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_fit = go.Figure()
                        
                        colors = []
                        for score in fit_df['Score_Numeric']:
                            if score >= 0.7:
                                colors.append('#28a745')
                            elif score >= 0.5:
                                colors.append('#ffc107')
                            elif score >= 0.3:
                                colors.append('#fd7e14')
                            else:
                                colors.append('#dc3545')
                        
                        fig_fit.add_trace(go.Bar(
                            y=fit_df['Model'],
                            x=fit_df['Fit Score'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=fit_df['Fit Score'].apply(lambda x: f'{x:.2f}'),
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Fit Score: %{x:.2f}<extra></extra>'
                        ))
                        
                        fig_fit.add_vline(
                            x=min_fit_score,
                            line_dash="dash",
                            line_color="orange",
                            line_width=2,
                            annotation_text=f"{t('target')} ({min_fit_score:.1f})",
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
                        st.metric(
                            t('models_selected'),
                            len(info['recommended']),
                            f"of {len(info['fit_scores'])}"
                        )
                        
                        st.markdown(t('score_legend'))
                        st.markdown(f"🟢 {t('highly_recommended')}")
                        st.markdown(f"🟡 {t('recommended')}")
                        st.markdown(f"🟠 {t('marginal')}")
                        st.markdown(f"🔴 {t('not_suitable')}")
                        
                        st.markdown("---")
                        
                        avg_score = sum(info['fit_scores'].values()) / len(info['fit_scores'])
                        st.metric(t('avg_fit_score'), f"{avg_score:.2f}")
                    
                    st.markdown("---")
                    st.markdown(f"#### {t('detailed_analysis')}")
                    
                    if info['recommended']:
                        st.markdown(t('selected_models'))
                        
                        for model in info['recommended']:
                            model_display = MODEL_DISPLAY_NAMES.get(model, model)
                            score = info['fit_scores'][model]
                            
                            with st.container():
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.markdown(f"**{model_display}**")
                                with col_b:
                                    if score >= 0.7:
                                        st.success(f"{t('score')}: {score:.2f}")
                                    else:
                                        st.warning(f"{t('score')}: {score:.2f}")
                                
                                if model in info['explanations']:
                                    exp = info['explanations'][model]
                                    
                                    if exp['pass']:
                                        st.markdown(t('strengths'))
                                        for reason in exp['pass']:
                                            st.markdown(f"✓ {reason}")
                                    
                                    if exp['fail']:
                                        st.markdown(t('considerations'))
                                        for reason in exp['fail']:
                                            st.markdown(f"⚠ {reason}")
                                
                                st.markdown("")
                    
                    if show_all_scores and info['exclusions']:
                        st.markdown("---")
                        st.markdown(t('excluded_models'))
                        
                        for model, reason in info['exclusions'].items():
                            model_display = MODEL_DISPLAY_NAMES.get(model, model)
                            score = info['fit_scores'].get(model, 0.0)
                            
                            with st.expander(f"{model_display} (Fit: {score:.2f})"):
                                st.error(f"{t('primary_reason')} {reason}")
                                
                                if model in info['explanations']:
                                    exp = info['explanations'][model]
                                    
                                    if exp['fail']:
                                        st.markdown(t('issues'))
                                        for detail in exp['fail']:
                                            st.markdown(f"• {detail}")
                                    
                                    if exp['pass']:
                                        st.markdown(t('positive_factors'))
                                        for detail in exp['pass']:
                                            st.markdown(f"• {detail}")
            
            # AI Mode - Explain model selection for beginners
            if use_ai_mode and visual_explainer:
                st.markdown("#### 🎓 Why These Models? (Beginner Guide)")
                
                for ticker, info in model_selection_info.items():
                    explanation = visual_explainer.explain_model_selection(
                        ticker=ticker,
                        selected_models=[MODEL_DISPLAY_NAMES.get(m, m) for m in info['recommended']],
                        excluded_models=[MODEL_DISPLAY_NAMES.get(m, m) for m in info['exclusions'].keys()],
                        reasons=info['exclusions']
                    )
                    
                    if explanation:
                        with st.expander(f"💡 Why these models for {ticker}?"):
                            st.info(explanation)
        
        st.markdown("---")
        
        # Margin of Safety Analysis
        if margin_results:
            st.markdown(f"### {t('margin_analysis')}")
            
            df_margin = pd.DataFrame(margin_results)
            df_margin_display = df_margin.set_index(["Ticker", "Model"])
            
            format_dict_margin = {
                "Intrinsic Value": "${:,.2f}",
                t('current_price'): "${:,.2f}",
                "Target Buy Price": "${:,.2f}"
            }
            
            st.dataframe(
                df_margin_display.style.format(format_dict_margin).applymap(
                    lambda x: 'background-color: #d4edda' if x == "✓ Undervalued" else 'background-color: #f8d7da',
                    subset=['Status']
                ),
                use_container_width=True
            )
            
            st.markdown(f"#### {t('margin_by_model')}")
            
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
                title=t('margin_analysis'),
                labels={"Margin of Safety (%)": "Margin of Safety (%)"}
            )
            
            fig2.add_hline(
                y=margin_of_safety_pct * 100,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"{t('target')}: {margin_of_safety_pct*100:.0f}%"
            )
            
            fig2.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            fig2.update_layout(height=400, showlegend=True)
            fig2.for_each_xaxis(lambda axis: axis.update(tickangle=45))
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # AI Mode - Explain margin of safety for beginners
            if use_ai_mode and visual_explainer:
                st.markdown("#### 🎓 Understanding Margin of Safety (Beginner Guide)")
                
                for ticker in df_margin['Ticker'].unique():
                    ticker_margins = df_margin[df_margin['Ticker'] == ticker]
                    avg_margin = ticker_margins['Margin of Safety (%)'].mean()
                    has_good_margin = any('✓' in status for status in ticker_margins['Status'])
                    
                    explanation = visual_explainer.explain_margin_of_safety(
                        ticker=ticker,
                        margin_pct=avg_margin,
                        target_margin=margin_of_safety_pct * 100,
                        is_undervalued=has_good_margin
                    )
                    
                    if explanation:
                        with st.expander(f"💡 What's a good deal for {ticker}?"):
                            st.info(explanation)
                            
                            if has_good_margin:
                                st.success(f"✅ {ticker} currently offers a {avg_margin:.1f}% safety cushion - meets the {margin_of_safety_pct*100:.0f}% target!")
                            else:
                                st.warning(f"⚠️ {ticker}'s {avg_margin:.1f}% safety cushion is below the {margin_of_safety_pct*100:.0f}% target. Consider waiting for a better price.")
        
        # Sensitivity Analysis
        if enable_sensitivity and len(tickers) == 1:
            st.markdown("---")
            st.markdown(f"### {t('sensitivity_analysis')}")
            
            ticker = tickers[0]
            try:
                df_data = fetch_stock_data(ticker)
                df_iv = IVSimplifier(df_data).simplify()
                vc = ValuationCalculator(df_iv)
                
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
                
                st.info(t('sensitivity_info', ticker=ticker, param=sensitivity_param.lower()))
                
            except Exception as e:
                st.error(f"{t('sensitivity_failed')}: {str(e)}")
        
        elif enable_sensitivity and len(tickers) > 1:
            st.warning(t('sensitivity_single_only'))
        
        # Export Options
        st.markdown("---")
        st.markdown(f"### {t('export')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_iv = df_iv_results.to_csv(index=False)
            st.download_button(
                label=t('download_valuations'),
                data=csv_iv,
                file_name=f"intrinsic_values_{'-'.join(tickers)}.csv",
                mime="text/csv"
            )
        
        with col2:
            if margin_results:
                csv_margin = df_margin.to_csv(index=False)
                st.download_button(
                    label=t('download_margins'),
                    data=csv_margin,
                    file_name=f"margin_of_safety_{'-'.join(tickers)}.csv",
                    mime="text/csv"
                )
        
        with col3:
            report_text = f"""
INTRINSIC VALUE ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Tickers: {', '.join(tickers)}

PARAMETERS:
- Discount Rate: {discount_rate*100:.1f}%
- Terminal Growth: {terminal_growth*100:.1f}%
- Target Margin of Safety: {margin_of_safety_pct*100:.1f}%
- Selection Mode: {'AI Mode' if use_ai_mode else 'Auto-select' if use_smart_selection else 'Manual'}

INTRINSIC VALUES:
{df_iv_results.to_string()}

MARGIN OF SAFETY:
{df_margin.to_string() if margin_results else 'N/A'}
"""
            st.download_button(
                label=t('download_report'),
                data=report_text,
                file_name=f"valuation_report_{'-'.join(tickers)}.txt",
                mime="text/plain"
            )
    
    else:
        st.error(t('no_analysis'))
    
    if failed_tickers:
        st.markdown("---")
        st.error(f"{t('failed_analyze')}: {', '.join(failed_tickers)}")
        st.info(
            f"{t('troubleshooting')}\n"
            f"{t('verify_ticker')}\n"
            f"{t('check_history')}\n"
            f"{t('reit_warning')}\n"
            f"{t('try_later')}"
        )

# Information Section
st.markdown("---")

with st.expander(t('about_models')):
    if st.session_state.language == 'en':
        st.markdown("""
    ### Valuation Models Explained
    
    **1. Discounted Cash Flow (DCF)**
    - Projects future free cash flows and discounts them to present value
    - Best for: Companies with stable, positive cash flows
    - **Auto-Selected When:** Company has positive FCF in 60%+ of years
    
    **2. Dividend Discount Model (DDM)**
    - Values stock based on present value of future dividends
    - **Single-Stage:** For stable dividend payers (growth <8%)
    - **Multi-Stage:** For growing dividend payers (growth ≥8%)
    - **Auto-Selected When:** Company pays dividends consistently
    
    **3. P/E Multiplier Model**
    - Uses average historical P/E ratio × current EPS
    - Best for: Stable, profitable companies
    - **Auto-Selected When:** Positive earnings in 60%+ of years
    
    **4. Asset-Based Valuation**
    - Book value (Total Assets - Total Liabilities) per share
    - **Auto-Selected When:** Low profitability but strong asset base
    
    **5. Modern Graham Formula**
    - Benjamin Graham's formula adjusted for bond yields
    - **Auto-Selected When:** 5+ years of profitability, strong ROE, low debt
    """)

with st.expander(t('faq')):
    st.markdown("""
    **Q: What's the difference between the three modes?**  
    A: 
    - **Manual**: You choose models and set parameters
    - **Auto-select**: AI chooses models, you set parameters
    - **AI Mode**: AI chooses models AND optimizes parameters per company
    
    **Q: Why do different models give different values?**  
    A: Each model uses different assumptions. Look at the range and average.
    
    **Q: What is margin of safety?**  
    A: It's like a "safety cushion" - how much cheaper the stock is than its estimated value.
    """)

with st.expander(t('technical')):
    st.markdown(f"""
    ### Default Parameters
    - **Discount Rate**: {DEFAULT_DISCOUNT_RATE*100:.1f}%
    - **Terminal Growth Rate**: {DEFAULT_TERMINAL_GROWTH*100:.1f}%
    - **Margin of Safety**: {DEFAULT_MARGIN_OF_SAFETY*100:.0f}%
    - **Data Source**: Yahoo Finance via yfinance library
    
    ### AI Mode Features
    - Company-specific discount rates (6-20%)
    - Industry-adjusted parameters
    - Risk-based optimization
    - Beginner-friendly explanations
    """)

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: gray;'>{t('disclaimer')}</p>",
    unsafe_allow_html=True
)