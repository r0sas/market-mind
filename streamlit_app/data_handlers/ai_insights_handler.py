from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def process_ai_insights(
    ticker: str,
    valuation_result: Dict,
    quality_report: Dict,
    ai_generator: object
) -> Optional[str]:
    """
    Generate AI insights for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        valuation_result: Valuation processing results
        quality_report: Data quality report
        ai_generator: AI insights generator instance
        
    Returns:
        AI-generated insight text or None
    """
    try:
        from core.oldscripts.DataFetcher import DataFetcher
        
        # Get company info
        fetcher = DataFetcher(ticker)
        company_info = fetcher.get_info()
        sector = company_info.get('sector', 'Unknown')
        
        # Generate insights
        insight = ai_generator.generate_insights(
            ticker=ticker,
            current_price=valuation_result['current_price'],
            valuations=valuation_result['iv_values'],
            sector=sector,
            warnings=quality_report.get('warnings', []),
            confidence_scores=valuation_result['confidence_scores']
        )
        
        if insight:
            logger.info(f"Generated AI insights for {ticker}")
            return insight
        else:
            logger.warning(f"No AI insights generated for {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to generate AI insights for {ticker}: {e}", exc_info=True)
        return None

