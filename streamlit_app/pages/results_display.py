import streamlit as st
from typing import Dict
from translations.translator import get_text
from pages.ai_insights_display import display_ai_insights
from pages.competitive_display import display_competitive_comparison
from pages.ml_predictions_display import display_ml_predictions
from pages.margin_of_safety_display import display_margin_of_safety
from pages.sensitivity_display import display_sensitivity_analysis
from pages.model_selection_display import display_model_selection_analysis
from visualizations.tables import display_iv_table, display_confidence_table
from visualizations.charts import create_iv_comparison_chart
from utils.export import export_to_csv, generate_text_report

def display_results(results: Dict, config: Dict):
    """Display all analysis results"""
    
    if not results['valuations']:
        st.error(get_text('no_analysis'))
        display_failed_tickers(results.get('failed', []))
        return
    
    st.success(f"{get_text('success')} {len(results['valuations'])} {get_text('stocks')}")
    
    # Display warnings if requested
    if config.get('show_warnings'):
        display_warnings(results.get('warnings', {}))
    
    st.markdown("---")
    
    # ML Predictions (if enabled)
    if config.get('enable_ml_prediction') and results.get('ml_predictions'):
        display_ml_predictions(results['ml_predictions'], results['valuations'])
        st.markdown("---")
    
    # Competitive Comparison (if enabled)
    if config.get('enable_competitive') and results.get('competitive'):
        display_competitive_comparison(results['competitive'])
        st.markdown("---")
    
    # AI Parameter Optimization (if AI mode)
    if config.get('use_ai_mode') and results.get('ai_parameters'):
        display_ai_parameters(results['ai_parameters'])
        st.markdown("---")
    
    # AI Insights (if enabled)
    if config.get('enable_ai_insights') and results.get('ai_insights'):
        display_ai_insights(results['ai_insights'])
        st.markdown("---")
    
    # Main Results
    st.markdown(f"### {get_text('iv_summary')}")
    display_iv_table(results['valuations'], config)
    
    # Confidence scores
    if config.get('show_confidence'):
        with st.expander(get_text('confidence_scores')):
            display_confidence_table(results['valuations'])
    
    # Visualization
    st.markdown(f"### {get_text('iv_comparison')}")
    fig = create_iv_comparison_chart(results['valuations'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Selection Analysis
    if (config.get('use_smart_selection') or config.get('use_ai_mode')) and results.get('model_info'):
        display_model_selection_analysis(results['model_info'], config)
        st.markdown("---")
    
    # Margin of Safety
    if results.get('margins'):
        display_margin_of_safety(results['margins'], config)
        st.markdown("---")
    
    # Sensitivity Analysis
    if config.get('enable_sensitivity'):
        display_sensitivity_analysis(
            st.session_state.get('tickers', []),
            config
        )
        st.markdown("---")
    
    # Export Options
    display_export_options(results, config)

def display_warnings(warnings: Dict):
    """Display data quality warnings"""
    if not any(warnings.values()):
        return
    
    st.markdown(f"### {get_text('data_warnings')}")
    for ticker, ticker_warnings in warnings.items():
        if ticker_warnings:
            with st.expander(f"{ticker} - {len(ticker_warnings)} {get_text('warnings')}"):
                for warning in ticker_warnings:
                    st.warning(warning)

def display_ai_parameters(ai_parameters: Dict):
    """Display AI-optimized parameters"""
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
    
    import pandas as pd
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
            
            st.markdown("---")

def display_failed_tickers(failed: list):
    """Display failed tickers with troubleshooting"""
    if not failed:
        return
    
    st.markdown("---")
    st.error(f"{get_text('failed_analyze')}: {', '.join(failed)}")
    st.info(
        f"{get_text('troubleshooting')}\n"
        f"{get_text('verify_ticker')}\n"
        f"{get_text('check_history')}\n"
        f"{get_text('reit_warning')}\n"
        f"{get_text('try_later')}"
    )

def display_export_options(results: Dict, config: Dict):
    """Display export buttons"""
    st.markdown(f"### {get_text('export')}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        import pandas as pd
        df_iv = pd.DataFrame(results['valuations'])
        csv_iv = export_to_csv(df_iv, 'valuations.csv')
        st.download_button(
            label=get_text('download_valuations'),
            data=csv_iv,
            file_name="intrinsic_values.csv",
            mime="text/csv"
        )
    
    with col2:
        if results.get('margins'):
            df_margin = pd.DataFrame(results['margins'])
            csv_margin = export_to_csv(df_margin, 'margins.csv')
            st.download_button(
                label=get_text('download_margins'),
                data=csv_margin,
                file_name="margin_of_safety.csv",
                mime="text/csv"
            )
    
    with col3:
        tickers = st.session_state.get('tickers', [])
        report_text = generate_text_report(results, config, tickers)
        st.download_button(
            label=get_text('download_report'),
            data=report_text,
            file_name="valuation_report.txt",
            mime="text/plain"
        )

