# ============================================
# OPTIMIZED ML DATA COLLECTION
# Using your existing DataFetcher + IVSimplifier
# ============================================

import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from core.DataFetcher import DataFetcher, DataFetcherError
from core.IVSimplifier import IVSimplifier, SimplifierError
from core.MultiCompanyAggregator import MultiCompanyAggregator

logger = logging.getLogger(__name__)

# ============================================
# STEP 1: Expand Config.py with ML-focused metrics
# ============================================

# Add this to your core/Config.py file:

ESSENTIAL_METRICS_ML = [
    # Original essential metrics (keep these)
    'Free Cash Flow',
    'Operating Cash Flow',
    'Investing Cash Flow',
    'Annual Dividends',
    'P/E Ratio',
    'Diluted EPS',
    'Basic EPS',
    'Share Price',
    'Total Assets',
    'Total Liabilities Net Minority Interest',
    'Total Equity Gross Minority Interest',
    'Basic Average Shares',
    'Diluted Average Shares',
    'Net Income',
    'Ticker',
    
    # ADDITIONAL METRICS FOR ML (add these)
    # Profitability
    'Operating Income',
    'Gross Profit',
    'EBITDA',
    'Total Revenue',
    
    # More Valuation
    'Enterprise Value',
    'Market Cap',
    
    # More Balance Sheet
    'Current Assets',
    'Current Liabilities',
    'Total Debt',
    'Long Term Debt',
    'Cash And Cash Equivalents',
    'Stockholders Equity',
    
    # More Cash Flow
    'Capital Expenditure',
    'Financing Cash Flow',
    
    # Company Info
    'Sector',
    'Industry',
]

# ============================================
# STEP 2: Create Enhanced Simplifier for ML
# ============================================

class MLDataSimplifier(IVSimplifier):
    """
    Extended IVSimplifier for ML purposes.
    Adds feature engineering on top of your existing simplifier.
    """
    
    def __init__(self, comprehensive_df: pd.DataFrame):
        """Initialize with comprehensive data."""
        # Use expanded metrics list
        super().__init__(
            comprehensive_df,
            essential_metrics=ESSENTIAL_METRICS_ML,
            prioritize_recent=True
        )
        self.engineered_features = None
    
    def simplify_for_ml(self) -> pd.DataFrame:
        """
        Simplify data and add engineered features for ML.
        
        Returns:
            DataFrame with both raw and engineered features
        """
        # Use your existing simplify method
        simplified = self.simplify()
        
        # Add engineered features
        simplified = self._add_engineered_features(simplified)
        
        return simplified
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated features that ML models love.
        
        These are derived metrics that often have strong predictive power.
        """
        # Growth Rates (Year-over-Year)
        if 'Basic EPS' in df.index and len(df.columns) > 1:
            eps_values = df.loc['Basic EPS'].values
            eps_growth = [(eps_values[i] / eps_values[i+1] - 1) * 100 
                         if eps_values[i+1] != 0 else np.nan 
                         for i in range(len(eps_values)-1)]
            eps_growth.append(np.nan)  # No growth for oldest year
            df.loc['EPS Growth YoY'] = eps_growth
        
        if 'Total Revenue' in df.index and len(df.columns) > 1:
            rev_values = df.loc['Total Revenue'].values
            rev_growth = [(rev_values[i] / rev_values[i+1] - 1) * 100 
                         if rev_values[i+1] != 0 else np.nan 
                         for i in range(len(rev_values)-1)]
            rev_growth.append(np.nan)
            df.loc['Revenue Growth YoY'] = rev_growth
        
        if 'Free Cash Flow' in df.index and len(df.columns) > 1:
            fcf_values = df.loc['Free Cash Flow'].values
            fcf_growth = [(fcf_values[i] / fcf_values[i+1] - 1) * 100 
                         if fcf_values[i+1] != 0 else np.nan 
                         for i in range(len(fcf_values)-1)]
            fcf_growth.append(np.nan)
            df.loc['FCF Growth YoY'] = fcf_growth
        
        # Profitability Ratios
        if 'Net Income' in df.index and 'Total Revenue' in df.index:
            df.loc['Net Margin'] = (df.loc['Net Income'] / df.loc['Total Revenue']) * 100
        
        if 'Operating Income' in df.index and 'Total Revenue' in df.index:
            df.loc['Operating Margin'] = (df.loc['Operating Income'] / df.loc['Total Revenue']) * 100
        
        if 'Gross Profit' in df.index and 'Total Revenue' in df.index:
            df.loc['Gross Margin'] = (df.loc['Gross Profit'] / df.loc['Total Revenue']) * 100
        
        # Returns
        if 'Net Income' in df.index and 'Total Equity Gross Minority Interest' in df.index:
            equity = df.loc['Total Equity Gross Minority Interest']
            df.loc['ROE'] = (df.loc['Net Income'] / equity) * 100
        
        if 'Net Income' in df.index and 'Total Assets' in df.index:
            df.loc['ROA'] = (df.loc['Net Income'] / df.loc['Total Assets']) * 100
        
        # Leverage Ratios
        if 'Total Debt' in df.index and 'Total Equity Gross Minority Interest' in df.index:
            df.loc['Debt to Equity'] = df.loc['Total Debt'] / df.loc['Total Equity Gross Minority Interest']
        
        # Liquidity Ratios
        if 'Current Assets' in df.index and 'Current Liabilities' in df.index:
            df.loc['Current Ratio'] = df.loc['Current Assets'] / df.loc['Current Liabilities']
        
        if 'Cash And Cash Equivalents' in df.index and 'Current Liabilities' in df.index:
            cash_plus_current = df.loc['Cash And Cash Equivalents']
            if 'Current Assets' in df.index:
                # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
                # Approximation without inventory
                df.loc['Quick Ratio'] = cash_plus_current / df.loc['Current Liabilities']
        
        # Cash Flow Metrics
        if 'Free Cash Flow' in df.index and 'Total Revenue' in df.index:
            df.loc['FCF Margin'] = (df.loc['Free Cash Flow'] / df.loc['Total Revenue']) * 100
        
        if 'Free Cash Flow' in df.index and 'Market Cap' in df.index:
            df.loc['FCF Yield'] = (df.loc['Free Cash Flow'] / df.loc['Market Cap']) * 100
        
        # Dividend Metrics
        if 'Annual Dividends' in df.index and 'Basic EPS' in df.index:
            eps = df.loc['Basic EPS']
            df.loc['Payout Ratio'] = (df.loc['Annual Dividends'] / eps) * 100
        
        if 'Annual Dividends' in df.index and 'Share Price' in df.index:
            df.loc['Dividend Yield'] = (df.loc['Annual Dividends'] / df.loc['Share Price']) * 100
        
        # Valuation Metrics
        if 'Market Cap' in df.index and 'Total Revenue' in df.index:
            df.loc['Price to Sales'] = df.loc['Market Cap'] / df.loc['Total Revenue']
        
        if 'Market Cap' in df.index and 'Total Equity Gross Minority Interest' in df.index:
            df.loc['Price to Book'] = df.loc['Market Cap'] / df.loc['Total Equity Gross Minority Interest']
        
        if 'Enterprise Value' in df.index and 'EBITDA' in df.index:
            ebitda = df.loc['EBITDA']
            df.loc['EV to EBITDA'] = df.loc['Enterprise Value'] / ebitda
        
        # Quality Score (composite metric)
        if all(m in df.index for m in ['ROE', 'Net Margin', 'ROA']):
            roe = df.loc['ROE'].fillna(0)
            margin = df.loc['Net Margin'].fillna(0)
            roa = df.loc['ROA'].fillna(0)
            df.loc['Quality Score'] = (roe + margin + roa) / 3
        
        return df


# ============================================
# STEP 3: Enhanced Aggregator for ML
# ============================================

def collect_ml_training_data(
    tickers: List[str],
    output_file: str = 'ml_training_data.csv',
    use_concurrent: bool = True,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Collect comprehensive ML training data for multiple companies.
    
    Args:
        tickers: List of stock tickers
        output_file: Where to save the data
        use_concurrent: Whether to fetch in parallel
        max_workers: Number of concurrent workers
        
    Returns:
        Combined DataFrame ready for ML training
    """
    print(f"Collecting ML training data for {len(tickers)} companies...")
    print("This will take 10-30 minutes depending on your connection...\n")
    
    # Initialize aggregator with ML-focused simplifier
    aggregator = MultiCompanyAggregator(
        DataFetcher,
        MLDataSimplifier,  # Use enhanced simplifier
        max_workers=max_workers,
        delay=0.5
    )
    
    # Fetch all company data
    print("Step 1/3: Fetching financial data...")
    aggregator.fetch_multiple_companies(tickers, use_concurrent=use_concurrent)
    
    # Create combined DataFrame with all features
    print("\nStep 2/3: Creating combined dataset...")
    combined_df = aggregator.create_combined_dataframe(
        metrics=None,  # Use all available metrics from MLDataSimplifier
        latest_year_only=False  # Keep all years for time-series
    )
    
    # Add sequential year column
    print("Step 3/3: Adding sequential years and final processing...")
    combined_df = add_sequential_years(combined_df)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Data saved to {output_file}")
    print(f"   Shape: {combined_df.shape}")
    print(f"   Columns: {len(combined_df.columns)}")
    print(f"   Companies: {combined_df['Ticker'].nunique()}")
    print(f"   Sectors: {combined_df['Sector'].nunique()}")
    
    return combined_df


def add_sequential_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Sequential_Year column (1, 2, 3, ...) for each company.
    
    This is critical for time-series ML models.
    """
    df = df.sort_values(['Ticker', 'Year'])
    df['Sequential_Year'] = df.groupby('Ticker').cumcount() + 1
    return df


# ============================================
# STEP 4: Quick Data Quality Check
# ============================================

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Quick analysis of the collected data quality.
    
    Returns:
        Dict with quality metrics
    """
    report = {
        'total_rows': len(df),
        'total_companies': df['Ticker'].nunique(),
        'sectors': df['Sector'].unique().tolist(),
        'year_range': f"{df['Year'].min():.0f} - {df['Year'].max():.0f}",
        'avg_years_per_company': df.groupby('Ticker')['Year'].count().mean(),
        'missing_data_pct': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
    }
    
    # Check feature completeness
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    completeness = {}
    for col in numeric_cols:
        completeness[col] = (1 - df[col].isna().sum() / len(df)) * 100
    
    report['feature_completeness'] = completeness
    report['features_above_80pct'] = [k for k, v in completeness.items() if v > 80]
    
    return report


# ============================================
# USAGE EXAMPLE
# ============================================

def main():
    """
    Complete example: Collect ML training data from S&P 500
    """
    # Get S&P 500 tickers (use your existing function)
    import requests
    
    def get_sp500_tickers():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.text)
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    
    # OPTION 1: Test with small sample first
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                   'JPM', 'JNJ', 'PG', 'XOM', 'CVX']
    
    print("="*60)
    print("ML DATA COLLECTION - TEST RUN")
    print("="*60)
    
    test_df = collect_ml_training_data(
        tickers=test_tickers,
        output_file='ml_training_data_test.csv',
        use_concurrent=True,
        max_workers=3
    )
    
    # Analyze quality
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    quality = analyze_data_quality(test_df)
    
    print(f"Total rows: {quality['total_rows']}")
    print(f"Companies: {quality['total_companies']}")
    print(f"Sectors: {', '.join(quality['sectors'])}")
    print(f"Year range: {quality['year_range']}")
    print(f"Avg years/company: {quality['avg_years_per_company']:.1f}")
    print(f"Missing data: {quality['missing_data_pct']:.1f}%")
    
    print(f"\nFeatures with >80% completeness ({len(quality['features_above_80pct'])}):")
    for feature in quality['features_above_80pct'][:20]:  # Show first 20
        pct = quality['feature_completeness'][feature]
        print(f"  ✓ {feature}: {pct:.1f}%")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    print(test_df.head(10))
    
    # OPTION 2: Full S&P 500 (uncomment when ready)
    """
    print("\n\nReady to collect full S&P 500 data?")
    response = input("This will take 20-30 minutes. Continue? (y/n): ")
    
    if response.lower() == 'y':
        sp500_tickers = get_sp500_tickers()
        
        full_df = collect_ml_training_data(
            tickers=sp500_tickers,
            output_file='sp500_ml_training_data.csv',
            use_concurrent=True,
            max_workers=5
        )
        
        quality_full = analyze_data_quality(full_df)
        print(f"\n✅ Full dataset collected!")
        print(f"   Total samples: {quality_full['total_rows']}")
        print(f"   Companies: {quality_full['total_companies']}")
    """


if __name__ == "__main__":
    main()


# ============================================
# INTEGRATION WITH EXISTING WORKFLOW
# ============================================

"""
WORKFLOW:

1. Run this script to collect data:
   python optimized_data_collection.py
   
2. Output: ml_training_data.csv with ~50 features
   
3. Then proceed to Phase 2 (EDA):
   - Use SectorFinancialAnalyzer from ml_strategy_framework.py
   - Identify top features per sector
   
4. Then Phase 3 (Training):
   - Use SectorPredictiveModel from ml_strategy_framework.py
   - Train sector-specific models
   
5. Then Phase 4 (Integration):
   - Use StockPricePredictionDashboard
   - Integrate into Streamlit

ADVANTAGES OF THIS APPROACH:
✅ Reuses your existing, tested code
✅ No need to modify IVSimplifier or DataFetcher
✅ Adds engineered features automatically
✅ Works with MultiCompanyAggregator
✅ Same data structure as your current workflow
✅ Easy to integrate with Phase 2-4
"""