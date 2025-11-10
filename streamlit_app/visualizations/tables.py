import streamlit as st
import pandas as pd
from typing import List, Dict
from utils.formatting import format_currency

def display_iv_table(valuations: List[Dict], config: Dict):
    """Display intrinsic value comparison table"""
    df = pd.DataFrame(valuations)
    
    # Separate value and confidence columns
    value_cols = [col for col in df.columns if not col.endswith('_Confidence')]
    
    # Set index and prepare display
    df_display = df[value_cols].set_index('Ticker')
    
    # Format dict for currency columns
    exclude_format = ['Years of Data', 'ml_prediction', 'ticker']
    format_dict = {
        col: "${:,.2f}"
        for col in df_display.columns
        if col not in exclude_format
    }
    
    st.dataframe(
        df_display.style.format(format_dict).background_gradient(
            subset=[c for c in df_display.columns if c not in exclude_format],
            cmap='RdYlGn',
            axis=1
        ),
        use_container_width=True
    )

def display_margin_table(margin_data: List[Dict]):
    """Display margin of safety table"""
    df = pd.DataFrame(margin_data)
    df_display = df.set_index(['Ticker', 'Model'])
    
    format_dict = {
        "Intrinsic Value": "${:,.2f}",
        "Current Price": "${:,.2f}",
        "Target Buy Price": "${:,.2f}"
    }
    
    st.dataframe(
        df_display.style.format(format_dict).applymap(
            lambda x: 'background-color: #d4edda' if x == "✓ Undervalued" else 'background-color: #f8d7da',
            subset=['Status']
        ),
        use_container_width=True
    )

def display_confidence_table(valuations: List[Dict]):
    """Display confidence scores table"""
    df = pd.DataFrame(valuations)
    
    # Extract confidence columns
    confidence_cols = [col for col in df.columns if col.endswith('_Confidence')]
    
    if not confidence_cols:
        st.warning("No confidence scores available")
        return
    
    confidence_data = []
    for _, row in df.iterrows():
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

