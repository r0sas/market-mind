# display_summary.py

import logging

def display_summary(df):
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Print available metrics and data quality warnings
    logger.info("Available Metrics: " + ", ".join(df['Year'].unique()))
