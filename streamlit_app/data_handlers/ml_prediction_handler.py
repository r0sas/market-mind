import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def process_ml_predictions(ticker: str, df_iv: pd.DataFrame, current_price: float) -> Optional[Dict]:
    """
    Process ML predictions for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        df_iv: Simplified financial data
        current_price: Current stock price
        
    Returns:
        Dict with ML prediction or None
    """
    try:
        # Clear any cached imports
        import importlib
        import sys
        
        if 'ml_models.stock_predictor' in sys.modules:
            del sys.modules['ml_models.stock_predictor']
        if 'ml_models' in sys.modules:
            del sys.modules['ml_models']
        
        # Fresh import
        from ml_models.stock_predictor import StockPricePredictionDashboard
        from core.oldscripts.DataFetcher import DataFetcher
        
        # Initialize dashboard
        dashboard = StockPricePredictionDashboard(models_dir='models/')
        
        # Load models
        if not dashboard.load_sector_models():
            logger.warning("ML models failed to load")
            return None
        
        # Get company sector
        temp_fetcher = DataFetcher(ticker)
        company_info = temp_fetcher.get_info()
        sector = company_info.get('sector', 'Unknown')
        
        # Prepare latest metrics for prediction
        latest_metrics = {}
        
        # Extract metrics from df_iv
        metric_mapping = {
            'Basic EPS': 'Basic EPS',
            'Free Cash Flow': 'Free Cash Flow',
            'P/E Ratio': 'P/E Ratio',
            'Total Revenue': 'Total Revenue',
            'Net Income': 'Net Income',
            'Total Assets': 'Total Assets',
            'Total Liabilities': 'Total Liabilities',
            'Dividends': 'Dividends'
        }
        
        for key, metric in metric_mapping.items():
            if metric in df_iv.index and len(df_iv.columns) > 0:
                try:
                    latest_metrics[key] = float(df_iv.loc[metric, df_iv.columns[0]])
                except (ValueError, TypeError):
                    latest_metrics[key] = 0
            else:
                latest_metrics[key] = 0
        
        latest_metrics['Share Price'] = current_price
        
        # Create DataFrame
        metrics_df = pd.DataFrame([latest_metrics])
        
        # Get prediction
        prediction = dashboard.predict_stock_movement(
            ticker=ticker,
            company_data=metrics_df,
            sector=sector
        )
        
        logger.info(f"ML prediction for {ticker}: {prediction}")
        
        return prediction
        
    except ImportError as e:
        logger.error(f"ML module import failed: {e}")
        return None
    except Exception as e:
        logger.error(f"ML prediction failed for {ticker}: {e}", exc_info=True)
        return None
