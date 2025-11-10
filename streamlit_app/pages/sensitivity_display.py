import streamlit as st
from typing import List, Dict
from translations.translator import get_text
from data_handlers.stock_fetcher import fetch_stock_data
from core.simplifier.iv_simplifier import IVSimplifier
from core.valuation.valuation_manager import ValuationManager
from visualizations.charts import create_sensitivity_plot

def display_sensitivity_analysis(tickers: List[str], config: Dict):
    """Display sensitivity analysis"""
    
    if not config.get('enable_sensitivity'):
        return
    
    st.markdown(f"### {get_text('sensitivity_analysis')}")
    
    if len(tickers) > 1:
        st.warning(get_text('sensitivity_single_only'))
        return
    
    ticker = tickers[0]
    
    try:
        # Fetch and prepare data
        df_data = fetch_stock_data(ticker)
        df_iv = IVSimplifier(df_data).simplify()
        vc = ValuationManager(df_iv)
        
        # Parameter and model mapping
        param_map = {
            "Discount Rate": "discount_rate",
            "Terminal Growth": "terminal_growth",
            "Growth Rate": "growth_rate"
        }
        
        model_map = {
            "DCF": "dcf",
            "DDM Multi-Stage": "ddm_multi_stage"
        }
        
        sensitivity_model = config['sensitivity_model']
        sensitivity_param = config['sensitivity_param']
        
        # Calculate sensitivity
        sensitivity_data = vc.sensitivity_analysis(
            model=model_map[sensitivity_model],
            param=param_map[sensitivity_param],
            range_pct=0.3,
            steps=7
        )
        
        # Create plot
        fig = create_sensitivity_plot(
            sensitivity_data,
            f"{ticker} - {sensitivity_model}"
        )
        
        # Add current value marker
        if sensitivity_model == "DCF":
            current_val = vc.calculate_dcf(
                discount_rate=config['discount_rate'],
                terminal_growth_rate=config['terminal_growth']
            )
        else:
            current_val = vc.calculate_ddm()
        
        if current_val:
            base_param = config['discount_rate'] if sensitivity_param == "Discount Rate" else config['terminal_growth']
            import plotly.graph_objects as go
            fig.add_trace(go.Scatter(
                x=[base_param],
                y=[current_val],
                mode='markers',
                name='Current',
                marker=dict(size=15, color='red', symbol='star')
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(get_text('sensitivity_info', ticker=ticker, param=sensitivity_param.lower()))
        
    except Exception as e:
        st.error(f"{get_text('sensitivity_failed')}: {str(e)}")

