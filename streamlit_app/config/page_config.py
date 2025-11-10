import streamlit as st
from translations.translator import get_text

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=get_text('page_title'),
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

