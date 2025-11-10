import streamlit as st
from typing import Dict
from translations.translator import get_text

def display_ai_insights(insights: Dict):
    """Display AI-generated insights"""
    if not insights:
        return
    
    st.markdown(f"### {get_text('ai_analysis')}")
    st.markdown(get_text('ai_subtitle'))
    
    for ticker, insight in insights.items():
        with st.expander(f"💡 {ticker} - AI Analysis", expanded=True):
            st.markdown(insight)
            st.caption(get_text('ai_caption'))

