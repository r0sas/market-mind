import streamlit as st
from streamlit_app.translations.translator import get_text
from config.display_constants import DEFAULT_DISCOUNT_RATE, DEFAULT_TERMINAL_GROWTH, DEFAULT_MARGIN_OF_SAFETY
from config.sidebar_config import *
from components.ai_config import render_ai_config
from components.competitive_config import render_competitive_config
from components.ml_config import render_ml_config

def render_sidebar():
    """Render complete sidebar and return configuration"""
    st.sidebar.title(get_text('sidebar_title'))
    st.sidebar.markdown("---")
    
    # Model Selection
    config = render_model_selection()
    
    # Advanced Parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader(get_text('advanced_params'))
    
    config.update(render_dcf_parameters())
    config.update(render_analysis_options())
    
    # AI Configuration
    config.update(render_ai_config())
    
    # Competitive Comparison
    config.update(render_competitive_config())
    
    # ML Prediction
    config.update(render_ml_config())
    
    # Sensitivity Analysis
    config.update(render_sensitivity_options())
    
    render_sidebar_footer()
    
    return config

def render_model_selection():
    """Render model selection mode"""
    st.sidebar.subheader(get_text('model_selection_mode'))
    
    selection_mode = st.sidebar.radio(
        get_text('model_selection_question'),
        options=[get_text('auto_select'), get_text('manual_select'), "🤖 AI Mode (Smart Parameters)"],
        help=HELP_TEXTS['auto_select']
    )
    
    config = {
        'use_smart_selection': selection_mode == get_text('auto_select'),
        'use_ai_mode': selection_mode == "🤖 AI Mode (Smart Parameters)",
        'selected_models': None
    }
    
    if config['use_smart_selection'] or config['use_ai_mode']:
        st.sidebar.success("✨ Smart model selection enabled")
        config['min_fit_score'] = st.sidebar.slider(
            get_text('min_fit_score'),
            FIT_SCORE_RANGE[0],
            FIT_SCORE_RANGE[1],
            DEFAULT_FIT_SCORE,
            0.1,
            help=HELP_TEXTS['fit_score']
        )
        config['show_all_scores'] = st.sidebar.checkbox(
            get_text('show_excluded'),
            value=DEFAULT_SHOW_ALL_SCORES
        )
        
        if config['use_ai_mode']:
            st.sidebar.info("💡 AI will optimize discount rate, terminal growth, and projection years")
    else:
        st.sidebar.info("📋 Manual model selection active")
        from core.config import MODEL_DISPLAY_NAMES
        all_models = list(MODEL_DISPLAY_NAMES.values())
        config['selected_models'] = st.sidebar.multiselect(
            get_text('select_models'),
            all_models,
            default=all_models
        )
    
    return config

def render_dcf_parameters():
    """Render DCF parameters"""
    with st.sidebar.expander(get_text('dcf_params')):
        discount_rate = st.slider(
            get_text('discount_rate'),
            DISCOUNT_RATE_RANGE[0],
            DISCOUNT_RATE_RANGE[1],
            DEFAULT_DISCOUNT_RATE * 100,
            0.5
        ) / 100
        
        terminal_growth = st.slider(
            get_text('terminal_growth'),
            TERMINAL_GROWTH_RANGE[0],
            TERMINAL_GROWTH_RANGE[1],
            DEFAULT_TERMINAL_GROWTH * 100,
            0.25
        ) / 100
    
    return {'discount_rate': discount_rate, 'terminal_growth': terminal_growth}

def render_analysis_options():
    """Render analysis options"""
    with st.sidebar.expander(get_text('analysis_options')):
        show_confidence = st.checkbox(
            get_text('show_confidence'),
            value=DEFAULT_SHOW_CONFIDENCE
        )
        show_warnings = st.checkbox(
            get_text('show_warnings'),
            value=DEFAULT_SHOW_WARNINGS
        )
        use_weighted_avg = st.checkbox(
            get_text('weighted_avg'),
            value=DEFAULT_USE_WEIGHTED_AVG,
            help=HELP_TEXTS['weighted_avg']
        )
        margin_of_safety = st.slider(
            get_text('margin_safety'),
            MARGIN_OF_SAFETY_RANGE[0],
            MARGIN_OF_SAFETY_RANGE[1],
            int(DEFAULT_MARGIN_OF_SAFETY * 100),
            5,
            help=HELP_TEXTS['margin_safety']
        ) / 100
    
    return {
        'show_confidence': show_confidence,
        'show_warnings': show_warnings,
        'use_weighted_avg': use_weighted_avg,
        'margin_of_safety': margin_of_safety
    }

def render_sensitivity_options():
    """Render sensitivity analysis options"""
    config = {'enable_sensitivity': False}
    
    with st.sidebar.expander(get_text('sensitivity')):
        enable = st.checkbox(get_text('enable_sensitivity'))
        config['enable_sensitivity'] = enable
        
        if enable:
            config['sensitivity_model'] = st.selectbox(
                get_text('model_analyze'),
                ["DCF", "DDM Multi-Stage"]
            )
            config['sensitivity_param'] = st.selectbox(
                get_text('param_vary'),
                ["Discount Rate", "Terminal Growth", "Growth Rate"]
            )
    
    return config

def render_sidebar_footer():
    """Render sidebar footer"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💡 **Tip:** Start with Auto-select mode to see which models work best for your stocks!",
        unsafe_allow_html=True
    )
