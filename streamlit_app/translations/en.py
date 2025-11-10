"""English translations"""

TRANSLATIONS_EN = {
    # Page titles
    'page_title': 'Intrinsic Value Calculator',
    'page_subtitle': 'Professional stock valuation analysis powered by AI',
    
    # Header
    'enter_tickers': 'Enter Stock Tickers',
    'ticker_placeholder': 'e.g., AAPL, MSFT, GOOGL',
    'ticker_help': 'Enter one or more stock tickers separated by commas',
    'calculate': 'Calculate Intrinsic Value',
    'enter_ticker_error': 'Please enter at least one ticker symbol',
    
    # Sidebar
    'sidebar_title': '⚙️ Configuration',
    'model_selection_mode': 'Model Selection Mode',
    'model_selection_question': 'How would you like to select models?',
    'auto_select': '✨ Auto-select (Recommended)',
    'manual_select': '📋 Manual Selection',
    'auto_select_help': 'AI analyzes company data and selects best models automatically',
    'smart_enabled': '✨ Smart model selection enabled',
    'min_fit_score': 'Minimum Fit Score',
    'min_fit_score_help': 'Higher scores = better model fit to company data',
    'show_excluded': 'Show Excluded Models',
    'show_excluded_help': 'Display models that didn\'t meet fit score threshold',
    'manual_active': '📋 Manual model selection active',
    'model_selection': 'Select Valuation Models',
    'select_models': 'Choose models to use',
    'select_models_help': 'Select one or more valuation models',
    
    # Parameters
    'advanced_params': 'Advanced Parameters',
    'dcf_params': 'DCF Parameters',
    'discount_rate': 'Discount Rate (%)',
    'discount_rate_help': 'Required rate of return (WACC)',
    'terminal_growth': 'Terminal Growth Rate (%)',
    'terminal_growth_help': 'Long-term growth rate',
    
    # Analysis options
    'analysis_options': 'Analysis Options',
    'show_confidence': 'Show Confidence Scores',
    'show_warnings': 'Show Data Warnings',
    'weighted_avg': 'Use Weighted Average',
    'weighted_avg_help': 'Weight models by confidence scores',
    'margin_safety': 'Target Margin of Safety (%)',
    'margin_safety_help': 'Safety buffer below intrinsic value',
    
    # AI
    'ai_insights': '🤖 AI Insights',
    'enable_ai': 'Enable AI Insights',
    'enable_ai_help': 'Get AI-powered analysis and recommendations',
    'groq_key': 'Groq API Key',
    'groq_key_help': 'Get free API key from console.groq.com',
    'api_provided': '✅ API key provided',
    'api_info': 'Get a free API key at console.groq.com',
    'ai_init_failed': 'AI initialization failed',
    
    # Results
    'processing': 'Processing',
    'fetching': 'Fetching data for',
    'success': 'Successfully analyzed',
    'stocks': 'stocks',
    'data_warnings': '⚠️ Data Quality Warnings',
    'warnings': 'warnings',
    
    # Tables
    'current_price': 'Current Price',
    'models_selected': 'Models Selected',
    'iv_summary': '📊 Intrinsic Value Summary',
    'iv_comparison': 'Intrinsic Value Comparison',
    'confidence_scores': '🎯 Confidence Scores',
    
    # Model analysis
    'model_analysis': '🔍 Model Selection Analysis',
    'model_analysis_subtitle': '*AI-powered model fit analysis*',
    'model_details': 'Model Details',
    'highly_recommended': '**Highly Recommended** (0.70+)',
    'recommended': '**Recommended** (0.50-0.69)',
    'marginal': '**Marginal** (0.30-0.49)',
    'not_suitable': '**Not Suitable** (<0.30)',
    'score_legend': '**Score Legend:**',
    'avg_fit_score': 'Average Fit Score',
    'detailed_analysis': 'Detailed Analysis',
    'selected_models': '✅ Selected Models',
    'score': 'Score',
    'strengths': '**Strengths:**',
    'considerations': '**Considerations:**',
    'excluded_models': '❌ Excluded Models',
    'primary_reason': '**Primary Reason:**',
    'issues': '**Issues:**',
    'positive_factors': '**Positive Factors:**',
    'target': 'Target',
    
    # Margin of safety
    'margin_analysis': '💰 Margin of Safety Analysis',
    'margin_by_model': 'Margin of Safety by Model',
    
    # Sensitivity
    'sensitivity': 'Sensitivity Analysis',
    'enable_sensitivity': 'Enable Sensitivity Analysis',
    'model_analyze': 'Model to Analyze',
    'param_vary': 'Parameter to Vary',
    'sensitivity_analysis': 'Sensitivity Analysis',
    'sensitivity_info': 'This chart shows how {ticker}\'s intrinsic value changes as {param} varies',
    'sensitivity_failed': 'Sensitivity analysis failed',
    'sensitivity_single_only': '⚠️ Sensitivity analysis only available for single ticker analysis',
    
    # Export
    'export': '📥 Export Results',
    'download_valuations': '📄 Download Valuations (CSV)',
    'download_margins': '📄 Download Margin Analysis (CSV)',
    'download_report': '📄 Download Full Report (TXT)',
    
    # AI Analysis
    'ai_analysis': '🤖 AI-Powered Analysis',
    'ai_subtitle': '*Generated insights based on valuation models and market data*',
    'ai_caption': '💡 AI-generated insight. Always conduct your own research.',
    
    # Errors
    'no_analysis': '❌ No stocks were successfully analyzed',
    'failed_analyze': '⚠️ Failed to analyze',
    'troubleshooting': '**Troubleshooting:**',
    'verify_ticker': '• Verify ticker symbols are correct',
    'check_history': '• Ensure stocks have sufficient historical data (5+ years)',
    'reit_warning': '• REITs may not work with all models',
    'try_later': '• Try again later if data source is unavailable',
    
    # Info sections
    'about_models': '📚 About Valuation Models',
    'faq': '❓ Frequently Asked Questions',
    'technical': '⚙️ Technical Details',
    'tip': '💡 **Tip:** Start with Auto-select mode to see which models work best!',
    
    # Disclaimer
    'disclaimer': '⚠️ This analysis is for informational purposes only. Not financial advice. Always consult a qualified advisor.'
}

