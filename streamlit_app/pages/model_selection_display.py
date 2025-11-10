import streamlit as st
from typing import Dict
from translations.translator import get_text
from visualizations.charts import create_fit_score_chart
from core.config import MODEL_DISPLAY_NAMES

def display_model_selection_analysis(model_info: Dict, config: Dict):
    """Display model selection analysis"""
    st.markdown(f"### {get_text('model_analysis')}")
    st.markdown(get_text('model_analysis_subtitle'))
    
    for ticker, info in model_info.items():
        with st.expander(f"📊 {ticker} - {get_text('model_details')}", expanded=False):
            
            # Create fit scores visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_fit_score_chart(
                    info['fit_scores'],
                    config['min_fit_score'],
                    ticker
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                display_fit_summary(info, config)
            
            st.markdown("---")
            st.markdown(f"#### {get_text('detailed_analysis')}")
            
            # Selected models
            if info['recommended']:
                display_selected_models(info)
            
            # Excluded models
            if config.get('show_all_scores') and info['exclusions']:
                display_excluded_models(info)

def display_fit_summary(info: Dict, config: Dict):
    """Display fit score summary metrics"""
    st.metric(
        get_text('models_selected'),
        len(info['recommended']),
        f"of {len(info['fit_scores'])}"
    )
    
    st.markdown(get_text('score_legend'))
    st.markdown(f"🟢 {get_text('highly_recommended')}")
    st.markdown(f"🟡 {get_text('recommended')}")
    st.markdown(f"🟠 {get_text('marginal')}")
    st.markdown(f"🔴 {get_text('not_suitable')}")
    
    st.markdown("---")
    
    avg_score = sum(info['fit_scores'].values()) / len(info['fit_scores'])
    st.metric(get_text('avg_fit_score'), f"{avg_score:.2f}")

def display_selected_models(info: Dict):
    """Display selected models with explanations"""
    st.markdown(get_text('selected_models'))
    
    for model in info['recommended']:
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        score = info['fit_scores'][model]
        
        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{model_display}**")
            with col_b:
                if score >= 0.7:
                    st.success(f"{get_text('score')}: {score:.2f}")
                else:
                    st.warning(f"{get_text('score')}: {score:.2f}")
            
            if model in info['explanations']:
                exp = info['explanations'][model]
                
                if exp['pass']:
                    st.markdown(get_text('strengths'))
                    for reason in exp['pass']:
                        st.markdown(f"✓ {reason}")
                
                if exp['fail']:
                    st.markdown(get_text('considerations'))
                    for reason in exp['fail']:
                        st.markdown(f"⚠ {reason}")
            
            st.markdown("")

def display_excluded_models(info: Dict):
    """Display excluded models with reasons"""
    st.markdown("---")
    st.markdown(get_text('excluded_models'))
    
    for model, reason in info['exclusions'].items():
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        score = info['fit_scores'].get(model, 0.0)
        
        with st.expander(f"{model_display} (Fit: {score:.2f})"):
            st.error(f"{get_text('primary_reason')} {reason}")
            
            if model in info['explanations']:
                exp = info['explanations'][model]
                
                if exp['fail']:
                    st.markdown(get_text('issues'))
                    for detail in exp['fail']:
                        st.markdown(f"• {detail}")
                
                if exp['pass']:
                    st.markdown(get_text('positive_factors'))
                    for detail in exp['pass']:
                        st.markdown(f"• {detail}")

