import streamlit as st
from translations.translator import get_text, get_available_languages

def render_language_selector():
    """Render language selector in top-right corner"""
    col1, col2 = st.columns([5, 1])
    with col2:
        languages = get_available_languages()
        selected = st.selectbox(
            '🌐',
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            index=list(languages.keys()).index(st.session_state.language),
            key='lang_selector',
            label_visibility='collapsed'
        )
        if selected != st.session_state.language:
            st.session_state.language = selected
            st.rerun()

def render_header():
    """Render page header with title and ticker input"""
    render_language_selector()
    
    st.title(f"📈 {get_text('page_title')}")
    st.markdown(get_text('page_subtitle'))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            get_text('enter_tickers'),
            value="AAPL",
            placeholder=get_text('ticker_placeholder'),
            help=get_text('ticker_help')
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(get_text('calculate'), type="primary", use_container_width=True):
            if ticker_input.strip():
                tickers = [t.strip().upper() for t in ticker_input.split(",")]
                st.session_state.tickers = list(dict.fromkeys(tickers))
                st.session_state.calculate_triggered = True
                st.rerun()
            else:
                st.error(get_text('enter_ticker_error'))

