import streamlit as st
from typing import Dict
from utils.formatting import create_confidence_badge, get_combined_signal

def display_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Display a metric card"""
    st.metric(label=label, value=value, delta=delta, help=help_text)

def display_confidence_badge(confidence: str):
    """Display confidence level badge"""
    badge_html = create_confidence_badge(confidence)
    st.markdown(badge_html, unsafe_allow_html=True)

def display_combined_signal(iv_diff_pct: float, ml_prediction: Dict):
    """Display combined IV + ML signal"""
    ml_direction = ml_prediction.get('prediction', 'FLAT ➡️')
    
    signal_data = get_combined_signal(iv_diff_pct, ml_direction)
    
    # Display signal with appropriate styling
    if signal_data['color'] == 'success':
        st.success(f"**{signal_data['signal']}**")
    elif signal_data['color'] == 'error':
        st.error(f"**{signal_data['signal']}**")
    elif signal_data['color'] == 'warning':
        st.warning(f"**{signal_data['signal']}**")
    else:
        st.info(f"**{signal_data['signal']}**")
    
    st.caption(signal_data['explanation'])

