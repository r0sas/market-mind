import streamlit as st
import pandas as pd
import logging
from typing import Dict, List
from core.datafetcher.data_fetcher import DataFetcher, DataFetcherError
from core.datafetcher.financial_data_processor import FinancialDataProcessor
from core.simplifier.iv_simplifier import IVSimplifier, SimplifierError
from streamlit_app.data_handlers.valuation_processor import process_valuations
from streamlit_app.data_handlers.competitive_handler import process_competitive_analysis
from streamlit_app.data_handlers.ml_prediction_handler import process_ml_predictions
from streamlit_app.data_handlers.ai_insights_handler import process_ai_insights

logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str):
    try:
        fetcher = FinancialDataProcessor(ticker)  # use the subclass with get_comprehensive_data
        df_data = fetcher.get_comprehensive_data()
        return df_data
    except Exception as e:
        logger.error(f"Unexpected error processing {ticker}: {e}")
        return None

def process_tickers(tickers: List[str], config: Dict) -> Dict:
    """
    Process all tickers and return comprehensive results.
    
    Args:
        tickers: List of stock ticker symbols
        config: Configuration dict from sidebar
        
    Returns:
        Dict containing all analysis results
    """
    results = {
        'valuations': [],
        'margins': [],
        'warnings': {},
        'model_info': {},
        'ai_insights': {},
        'ai_parameters': {},
        'competitive': {},
        'ml_predictions': {},
        'failed': []
    }
    
    # Initialize progress tracking
    progress = st.progress(0)
    status = st.empty()
    
    # Initialize AI components if enabled
    ai_generator, param_optimizer, visual_explainer = initialize_ai_components(config)
    
    for idx, ticker in enumerate(tickers):
        status.text(f"Processing {ticker}... ({idx + 1}/{len(tickers)})")
        
        try:
            # Fetch and simplify data
            with st.spinner(f"Fetching {ticker} data..."):
                df_data = fetch_stock_data(ticker)
                simplifier = IVSimplifier(df_data)
                df_iv = simplifier.simplify()
            
            # Get data quality report
            quality_report = simplifier.get_data_quality_report()
            results['warnings'][ticker] = quality_report.get('warnings', [])
            
            # Process valuations
            valuation_result = process_valuations(
                ticker, df_iv, config,
                param_optimizer if config.get('use_ai_mode') else None
            )
            
            results['valuations'].append(valuation_result['iv_data'])
            results['margins'].extend(valuation_result['margin_data'])
            results['model_info'][ticker] = valuation_result['model_info']
            
            if valuation_result.get('ai_parameters'):
                results['ai_parameters'][ticker] = valuation_result['ai_parameters']
            
            # Process competitive analysis
            if config.get('enable_competitive'):
                status.text(f"🔍 Detecting competitors for {ticker}...")
                comp_result = process_competitive_analysis(ticker, config)
                if comp_result:
                    results['competitive'][ticker] = comp_result
            
            # Process ML predictions
            if config.get('enable_ml_prediction'):
                status.text(f"🤖 Generating ML prediction for {ticker}...")
                ml_result = process_ml_predictions(ticker, df_iv, valuation_result['current_price'])
                if ml_result:
                    results['ml_predictions'][ticker] = ml_result
                    # Add to valuation data
                    results['valuations'][-1]['ml_prediction'] = ml_result
            
            # Process AI insights
            if config.get('enable_ai_insights') and ai_generator:
                status.text(f"💡 Generating AI insights for {ticker}...")
                insight = process_ai_insights(
                    ticker,
                    valuation_result,
                    quality_report,
                    ai_generator
                )
                if insight:
                    results['ai_insights'][ticker] = insight
            
        except (DataFetcherError, SimplifierError) as e:
            logger.error(f"Error processing {ticker}: {e}")
            st.error(f"❌ Error processing {ticker}: {str(e)}")
            results['failed'].append(ticker)
        except Exception as e:
            logger.error(f"Unexpected error processing {ticker}: {e}", exc_info=True)
            st.error(f"❌ Unexpected error with {ticker}: {str(e)}")
            results['failed'].append(ticker)
        
        progress.progress((idx + 1) / len(tickers))
    
    # Clear progress indicators
    progress.empty()
    status.empty()
    
    return results

def initialize_ai_components(config: Dict):
    """Initialize AI components if enabled"""
    ai_generator = None
    param_optimizer = None
    visual_explainer = None
    
    if config.get('enable_ai_insights') and (config.get('groq_api_key') or config.get('use_ollama')):
        try:
            from core.ai_insights.ai_insights import AIInsightsGenerator
            
            ai_generator = AIInsightsGenerator(
                api_key=config.get('groq_api_key'),
                use_ollama=config.get('use_ollama', False)
            )
            
            if ai_generator.test_connection():
                provider = "Ollama (Local)" if config.get('use_ollama') else "Groq API"
                st.info(f"✅ {provider} connected")
            else:
                st.warning("⚠️ AI connection failed. Using fallback.")
                ai_generator = None
        except Exception as e:
            logger.error(f"Failed to initialize AI insights: {e}")
            st.error(f"AI insights initialization failed: {e}")
            ai_generator = None
    
    if config.get('use_ai_mode'):
        try:
            from core.ai_optimizer.optimizer import AIParameterOptimizer
            from core.ai_visual_explainer.explainer import AIVisualExplainer
            
            # Use rule-based for Ollama (needs larger model)
            if config.get('use_ollama'):
                param_optimizer = AIParameterOptimizer(use_ai=False)
                st.info("📊 Using rule-based parameter optimization")
            else:
                param_optimizer = AIParameterOptimizer(
                    api_key=config.get('groq_api_key'),
                    use_ai=True
                )
            
            visual_explainer = AIVisualExplainer(
                api_key=config.get('groq_api_key'),
                use_ollama=config.get('use_ollama', False)
            )
            
            provider = "Ollama (Local)" if config.get('use_ollama') else "Groq API"
            st.info(f"🤖 AI features enabled ({provider})")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI mode: {e}")
            st.warning(f"⚠️ AI features unavailable: {e}")
            param_optimizer = AIParameterOptimizer(use_ai=False)
            visual_explainer = AIVisualExplainer()
    
    return ai_generator, param_optimizer, visual_explainer
