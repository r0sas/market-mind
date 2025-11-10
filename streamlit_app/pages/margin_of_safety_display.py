import streamlit as st
import pandas as pd
from typing import List, Dict
from translations.translator import get_text
from visualizations.charts import create_margin_chart

def display_margin_of_safety(margins: List[Dict], config: Dict):
    """Display margin of safety analysis"""
    st.markdown(f"### {get_text('margin_analysis')}")
    
    from visualizations.tables import display_margin_table
    display_margin_table(margins)
    
    st.markdown(f"#### {get_text('margin_by_model')}")
    
    fig = create_margin_chart(margins, config['margin_of_safety'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary by ticker
    display_margin_summary(margins, config['margin_of_safety'])

def display_margin_summary(margins: List[Dict], target_margin: float):
    """Display margin of safety summary"""
    df = pd.DataFrame(margins)
    
    with st.expander("📊 Margin of Safety Summary"):
        for ticker in df['Ticker'].unique():
            ticker_margins = df[df['Ticker'] == ticker]
            avg_margin = ticker_margins['Margin of Safety (%)'].mean()
            has_good_margin = any('✓' in status for status in ticker_margins['Status'])
            
            st.markdown(f"#### {ticker}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Margin", f"{avg_margin:.1f}%")
            with col2:
                st.metric("Target Margin", f"{target_margin*100:.0f}%")
            
            if has_good_margin:
                st.success(f"✅ {ticker} offers adequate safety margin!")
            else:
                st.warning(f"⚠️ {ticker} below target margin. Consider waiting for better price.")
            
            st.markdown("---")

