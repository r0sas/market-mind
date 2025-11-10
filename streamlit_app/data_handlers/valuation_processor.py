from typing import Dict, Optional
import pandas as pd
import logging
from core.valuation.base_valuation import ValuationError
from core.valuation.valuation_manager import ValuationManager
from core.model_selector.model_selector import ModelSelector
from core.datafetcher.data_fetcher import DataFetcher
from core.config import MODEL_DISPLAY_NAMES

logger = logging.getLogger(__name__)

def process_valuations(
    ticker: str,
    df_iv: pd.DataFrame,
    config: Dict,
    param_optimizer: Optional[object] = None
) -> Dict:
    """
    Process valuations for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        df_iv: Simplified financial data
        config: Configuration dict
        param_optimizer: AI parameter optimizer (optional)
        
    Returns:
        Dict with valuation results
    """
    # AI Mode - Optimize Parameters
    company_discount_rate = config['discount_rate']
    company_terminal_growth = config['terminal_growth']
    ai_parameters = None
    
    if config.get('use_ai_mode') and param_optimizer:
        try:
            temp_fetcher = DataFetcher(ticker)
            company_info = temp_fetcher.get_info()
            
            # Prepare financial metrics
            financial_metrics = {
                'beta': company_info.get('beta', 1.0),
                'debt_to_equity': company_info.get('debtToEquity', 0) / 100 if company_info.get('debtToEquity') else 0,
                'revenue_cagr': 0.10,  # Could calculate from df_iv
                'fcf_cagr': 0.08,
                'roe': company_info.get('returnOnEquity', 0),
                'net_margin': company_info.get('profitMargins', 0),
                'total_debt': company_info.get('totalDebt', 0),
                'interest_expense': 0,
                'tax_rate': 0.21
            }
            
            company_data = {
                'sector': company_info.get('sector', 'default'),
                'industry': company_info.get('industry', 'Unknown'),
                'marketCap': company_info.get('marketCap', 0)
            }
            
            # Optimize parameters
            optimized = param_optimizer.optimize_parameters(
                ticker, company_data, financial_metrics
            )
            
            ai_parameters = optimized
            company_discount_rate = optimized['discount_rate']
            company_terminal_growth = optimized['terminal_growth']
            
            logger.info(f"AI optimized params for {ticker}: DR={company_discount_rate:.2%}, TG={company_terminal_growth:.2%}")
            
        except Exception as e:
            logger.error(f"AI parameter optimization failed for {ticker}: {e}")
            # Fall back to default parameters
    
    # Model Selection Logic
    if config.get('use_smart_selection') or config.get('use_ai_mode'):
        # Initialize and analyze model fits
        selector = ModelSelector(df_iv)
        selector.analyze()
        
        fit_scores = selector.fit_scores
        explanations = selector.explanations
        recommended_models = selector.recommend(top_n=3)
        
        model_info = {
            'fit_scores': fit_scores,
            'recommended': recommended_models,
            'explanations': explanations,
            'selector': selector
        }
        
        # Fallback: ensure models are extracted from recommendations
        models = [m for m in recommended_models.keys()] if isinstance(recommended_models, dict) else recommended_models
        
        if not models:
            logger.warning(f"No suitable models found for {ticker}")
            raise ValuationError(
                f"No models met the fit criteria. Try checking data quality or parameters."
            )

    else:
        # Manual selection
        name_map = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
        models = [name_map[m] for m in config['selected_models'] if m in name_map]
        model_info = None
    
    # Calculate valuations
    vc = ValuationManager(df_iv)
    vc.calculate_models(
        models_to_calculate=models,
        discount_rate=company_discount_rate,
        terminal_growth_rate=company_terminal_growth
    )

    
    # Prepare intrinsic value results
    iv_data = {
        'Ticker': ticker,
        'ticker': ticker,  # lowercase for compatibility
        'current_price': vc.current_price,
        'average_iv': vc.get_average_valuation(config['use_weighted_avg'])
    }
    
    # Add model values
    for model, value in vc.get_results().items():
        if value and value > 0:
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            iv_data[display_name] = value
            
            # Add confidence scores if requested
            if config.get('show_confidence'):
                confidence_scores = vc.get_confidence_scores()
                if model in confidence_scores:
                    iv_data[f"{display_name}_Confidence"] = confidence_scores[model]
    
    # Prepare margin of safety results
    margin_data = []
    margin_analysis = vc.get_margin_of_safety(config['margin_of_safety'])
    
    for model, data in margin_analysis.items():
        if data['intrinsic_value'] and data['intrinsic_value'] > 0:
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            margin_data.append({
                'Ticker': ticker,
                'Model': display_name,
                'Intrinsic Value': data['intrinsic_value'],
                'Current Price': data['current_price'],
                'Margin of Safety (%)': round(data['margin_of_safety'] * 100, 1),
                'Target Buy Price': data['target_buy_price'],
                'Status': "✓ Undervalued" if data['is_undervalued'] else "✗ Overvalued",
                'Confidence': data.get('confidence', 'N/A')
            })
    
    return {
        'iv_data': iv_data,
        'margin_data': margin_data,
        'model_info': model_info,
        'current_price': vc.current_price,
        'iv_values': vc.get_results(),
        'confidence_scores': vc.get_confidence_scores(),
        'ai_parameters': ai_parameters
    }
