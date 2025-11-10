from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def process_competitive_analysis(ticker: str, config: Dict) -> Optional[Dict]:
    """
    Process competitive comparison for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        config: Configuration dict
        
    Returns:
        Dict with competitive analysis results or None
    """
    try:
        from core.oldscripts.competitive_comparison import CompetitiveComparison
        
        # Parse manual competitors if provided
        manual_comps = config.get('manual_competitors')
        
        # Initialize competitive comparison
        comparison = CompetitiveComparison(
            ticker=ticker,
            api_key=config.get('groq_api_key') if not config.get('use_ollama') else None,
            use_ollama=config.get('use_ollama', False),
            manual_competitors=manual_comps
        )
        
        # Detect competitors
        competitors = comparison.detect_competitors()
        
        if not competitors:
            logger.warning(f"Could not detect competitors for {ticker}")
            return None
        
        # Fetch all data
        comparison.fetch_all_data()
        
        # Generate comparison table
        comp_table = comparison.generate_comparison_table()
        
        # Generate charts
        price_chart_1y = comparison.create_price_comparison_chart(period='1y')
        price_chart_6m = comparison.create_price_comparison_chart(period='6mo')
        price_chart_3m = comparison.create_price_comparison_chart(period='3mo')
        metrics_radar = comparison.create_metrics_radar_chart()
        
        # Generate summary
        comp_summary = comparison.generate_summary()
        
        logger.info(f"✓ Generated competitive comparison for {ticker}")
        
        return {
            'competitors': competitors,
            'table': comp_table,
            'chart_1y': price_chart_1y,
            'chart_6m': price_chart_6m,
            'chart_3m': price_chart_3m,
            'radar': metrics_radar,
            'summary': comp_summary,
            'data': comparison.data
        }
        
    except Exception as e:
        logger.error(f"Competitive comparison failed for {ticker}: {e}", exc_info=True)
        return None

