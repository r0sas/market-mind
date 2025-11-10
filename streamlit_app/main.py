import streamlit as st
from config.page_config import setup_page_config
from components.header import render_header
from components.sidebar import render_sidebar
from data_handlers.stock_fetcher import process_tickers
from pages.results_display import display_results
from pages.information_sections import render_information_sections
from utils.session_state import initialize_session_state

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main application entry point"""
    # Initialize
    initialize_session_state()
    setup_page_config()
    
    # Render UI
    render_header()
    sidebar_config = render_sidebar()
    
    # Handle calculation
    if st.session_state.get('calculate_triggered'):
        tickers = st.session_state.get('tickers', [])
        if tickers:
            results = process_tickers(tickers, sidebar_config)
            display_results(results, sidebar_config)
        st.session_state.calculate_triggered = False
    
    # Information sections at bottom
    render_information_sections()

if __name__ == "__main__":
    main()