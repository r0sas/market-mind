# validate_data_quality.py

import logging

def validate_data_quality(df):
    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    
    # Check for missing years or essential metrics
    if 'Year' not in df.columns:
        logger.error("Data quality issue: 'Year' column is missing.")
    
    essential_metrics = ['Free Cash Flow', 'EPS', 'P/E Ratio']
    missing_metrics = set(essential_metrics) - set(df['Year'])
    for metric in missing_metrics:
        logger.warning(f"Missing essential metric: {metric}")
