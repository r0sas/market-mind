import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'language': 'en',
        'calculate_triggered': False,
        'tickers': [],
        'results': None,
        'ai_insights': {},
        'competitive_results': {},
        'ml_predictions': {},
        'last_config': {},
        'processing': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_state(key, value):
    """Update session state safely"""
    st.session_state[key] = value

def get_state(key, default=None):
    """Get session state value safely"""
    return st.session_state.get(key, default)

