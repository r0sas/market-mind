# ml_models/ml_data_collector.py
"""
ML Data Collection Script
Collects comprehensive financial data for machine learning training
"""

import pandas as pd
import numpy as np
import requests
from typing import List
import logging
from datetime import datetime

# Import your existing modules
import sys
sys.path.append('..')  # Go up one directory to access core/
from core.oldscripts.DataFetcher import DataFetcher
from core.IV_Simplifier.IV_Simplifier import IVSimplifier
from core.oldscripts.MultiCompanyAggregator import MultiCompanyAggregator
from core.Config import ESSENTIAL_METRICS_ML

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 ticker list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # yfinance compatibility
        print(f"✅ Found {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"❌ Failed to fetch S&P 500 tickers: {e}")
        return []


def add_sequential_years(df: pd.DataFrame) -> pd.DataFrame:
    """Add Sequential_Year column (1=oldest, 2, 3, ... N=newest)."""
    df = df.sort_values(['Ticker', 'Year'])
    df['Sequential_Year'] = df.groupby('Ticker').cumcount() + 1
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated features that improve ML predictions.
    These are ratios and growth rates derived from raw metrics.
    """
    print("Engineering features...")
    
    # Sort by ticker and year to ensure correct calculations
    df = df.sort_values(['Ticker', 'Year'])
    
    # === Growth Rates (Year-over-Year) ===
    for metric in ['Basic EPS', 'Total Revenue', 'Free Cash Flow', 'Net Income']:
        if metric in df.columns:
            df[f'{metric} Growth YoY'] = df.groupby('Ticker')[metric].pct_change() * 100
    
    # === Profitability Ratios ===
    if 'Net Income' in df.columns and 'Total Revenue' in df.columns:
        df['Net Margin'] = (df['Net Income'] / df['Total Revenue']) * 100
    
    if 'Operating Income' in df.columns and 'Total Revenue' in df.columns:
        df['Operating Margin'] = (df['Operating Income'] / df['Total Revenue']) * 100
    
    if 'Gross Profit' in df.columns and 'Total Revenue' in df.columns:
        df['Gross Margin'] = (df['Gross Profit'] / df['Total Revenue']) * 100
    
    # === Return Ratios ===
    if 'Net Income' in df.columns and 'Total Equity Gross Minority Interest' in df.columns:
        df['ROE'] = (df['Net Income'] / df['Total Equity Gross Minority Interest']) * 100
    
    if 'Net Income' in df.columns and 'Total Assets' in df.columns:
        df['ROA'] = (df['Net Income'] / df['Total Assets']) * 100
    
    # === Leverage Ratios ===
    if 'Total Debt' in df.columns and 'Total Equity Gross Minority Interest' in df.columns:
        df['Debt to Equity'] = df['Total Debt'] / df['Total Equity Gross Minority Interest']
    
    # === Liquidity Ratios ===
    if 'Current Assets' in df.columns and 'Current Liabilities' in df.columns:
        df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
    
    # === Cash Flow Metrics ===
    if 'Free Cash Flow' in df.columns and 'Total Revenue' in df.columns:
        df['FCF Margin'] = (df['Free Cash Flow'] / df['Total Revenue']) * 100
    
    if 'Free Cash Flow' in df.columns and 'Market Cap' in df.columns:
        df['FCF Yield'] = (df['Free Cash Flow'] / df['Market Cap']) * 100
    
    # === Dividend Metrics ===
    if 'Annual Dividends' in df.columns and 'Basic EPS' in df.columns:
        df['Payout Ratio'] = (df['Annual Dividends'] / df['Basic EPS']) * 100
    
    if 'Annual Dividends' in df.columns and 'Share Price' in df.columns:
        df['Dividend Yield'] = (df['Annual Dividends'] / df['Share Price']) * 100
    
    # === Valuation Ratios ===
    if 'Market Cap' in df.columns and 'Total Revenue' in df.columns:
        df['Price to Sales'] = df['Market Cap'] / df['Total Revenue']
    
    if 'Market Cap' in df.columns and 'Total Equity Gross Minority Interest' in df.columns:
        df['Price to Book'] = df['Market Cap'] / df['Total Equity Gross Minority Interest']
    
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print("✅ Feature engineering complete")
    return df


def collect_ml_data(
    tickers: List[str],
    output_file: str,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Collect comprehensive ML training data.
    
    Args:
        tickers: List of stock tickers
        output_file: Where to save CSV
        max_workers: Number of parallel workers
        
    Returns:
        DataFrame with all data
    """
    print(f"\n{'='*60}")
    print(f"ML DATA COLLECTION - {len(tickers)} COMPANIES")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    # Create custom IVSimplifier class that uses expanded metrics
    class MLSimplifier(IVSimplifier):
        def __init__(self, comprehensive_df):
            super().__init__(
                comprehensive_df,
                essential_metrics=ESSENTIAL_METRICS_ML,
                prioritize_recent=True
            )
    
    # Initialize aggregator
    aggregator = MultiCompanyAggregator(
        DataFetcher,
        MLSimplifier,
        max_workers=max_workers,
        delay=0.5
    )
    
    # Fetch all company data
    print("Step 1/4: Fetching financial data from Yahoo Finance...")
    print(f"Using {max_workers} parallel workers with 0.5s delay\n")
    
    aggregator.fetch_multiple_companies(tickers, use_concurrent=True)
    
    # Create combined DataFrame
    print("\nStep 2/4: Creating combined dataset...")
    combined_df = aggregator.create_combined_dataframe(
        metrics=ESSENTIAL_METRICS_ML,
        latest_year_only=False  # Keep all years
    )
    
    print(f"✅ Combined dataset: {combined_df.shape}")
    
    # Add sequential years
    print("\nStep 3/4: Adding sequential years...")
    combined_df = add_sequential_years(combined_df)
    
    # Engineer features
    print("\nStep 4/4: Engineering additional features...")
    combined_df = add_engineered_features(combined_df)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"\n{'='*60}")
    print(f"✅ DATA COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"File: {output_file}")
    print(f"Shape: {combined_df.shape}")
    print(f"Companies: {combined_df['Ticker'].nunique()}")
    print(f"Sectors: {combined_df['Sector'].nunique()}")
    print(f"Time: {elapsed:.1f} minutes")
    print(f"{'='*60}\n")
    
    return combined_df


def analyze_data_quality(df: pd.DataFrame):
    """Print data quality report."""
    print(f"\n{'='*60}")
    print("DATA QUALITY REPORT")
    print(f"{'='*60}\n")
    
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Companies: {df['Ticker'].nunique()}")
    print(f"Sectors: {df['Sector'].nunique()}")
    print(f"Year range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    print(f"Avg years per company: {df.groupby('Ticker')['Year'].count().mean():.1f}")
    
    # Missing data analysis
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    print(f"Missing data: {missing_pct:.1f}%")
    
    # Feature completeness
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    completeness = {}
    for col in numeric_cols:
        pct = (1 - df[col].isna().sum() / len(df)) * 100
        if pct > 80:  # Only show features with good completeness
            completeness[col] = pct
    
    print(f"\n✅ Features with >80% completeness: {len(completeness)}")
    print("\nTop 20 most complete features:")
    sorted_features = sorted(completeness.items(), key=lambda x: x[1], reverse=True)[:20]
    for feature, pct in sorted_features:
        print(f"  {feature:40s} {pct:5.1f}%")
    
    # Sector distribution
    print("\nCompanies per sector:")
    sector_counts = df.groupby('Sector')['Ticker'].nunique().sort_values(ascending=False)
    for sector, count in sector_counts.items():
        print(f"  {sector:40s} {count:3d} companies")


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("ML DATA COLLECTION WIZARD")
    print("="*60 + "\n")
    
    # Ask user what they want to do
    print("Options:")
    print("1. Test with 10 companies (2-3 minutes)")
    print("2. Test with 50 companies (10-15 minutes)")
    print("3. Full S&P 500 (~500 companies, 30-45 minutes)")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Test run with 10 companies
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                       'JPM', 'JNJ', 'PG', 'XOM', 'CVX']
        output_file = 'ml_data_test_10.csv'
        max_workers = 3
        
    elif choice == '2':
        # Medium test with 50 companies
        all_tickers = get_sp500_tickers()
        test_tickers = all_tickers[:50] if all_tickers else []
        output_file = 'ml_data_test_50.csv'
        max_workers = 5
        
    elif choice == '3':
        # Full S&P 500
        test_tickers = get_sp500_tickers()
        output_file = 'sp500_ml_training_data.csv'
        max_workers = 5
        
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not test_tickers:
        print("❌ No tickers to process. Exiting.")
        return
    
    print(f"\n✅ Will collect data for {len(test_tickers)} companies")
    print(f"Output file: {output_file}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Collect data
    df = collect_ml_data(
        tickers=test_tickers,
        output_file=output_file,
        max_workers=max_workers
    )
    
    # Analyze quality
    analyze_data_quality(df)
    
    # Show sample
    print(f"\n{'='*60}")
    print("SAMPLE DATA (first 10 rows)")
    print(f"{'='*60}\n")
    print(df.head(10).to_string())
    
    print(f"\n{'='*60}")
    print("✅ PHASE 1 COMPLETE!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Check the file: {output_file}")
    print(f"2. Review data quality")
    print(f"3. When ready, proceed to Phase 2 (EDA)")
    print()


if __name__ == "__main__":
    main()