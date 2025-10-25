"""
Test script for ModelSelector functionality.

Tests intelligent model selection with various company types:
- Dividend aristocrats (KO, JNJ)
- Growth stocks (NVDA, TSLA)
- Financial institutions (BAC, JPM)
- Cyclical companies (F, AAL)
- Tech giants (AAPL, MSFT)
"""

import sys
from core.DataFetcher import DataFetcher, DataFetcherError
from core.IVSimplifier import IVSimplifier, SimplifierError
from core.model_selector import ModelSelector
from core.ValuationCalculator import ValuationCalculator


def test_single_company(ticker: str, verbose: bool = True) -> dict:
    """
    Test model selection for a single company.
    
    Args:
        ticker: Stock ticker symbol
        verbose: If True, print detailed analysis
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*70}")
    print(f"TESTING: {ticker}")
    print('='*70)
    
    try:
        # Step 1: Fetch data
        print(f"\n[1/5] Fetching data for {ticker}...")
        fetcher = DataFetcher(ticker)
        comprehensive_data = fetcher.get_comprehensive_data()
        print(f"✓ Fetched {comprehensive_data.shape[0]} metrics across {comprehensive_data.shape[1]} periods")
        
        # Step 2: Simplify
        print(f"\n[2/5] Simplifying data...")
        simplifier = IVSimplifier(comprehensive_data)
        simplified_data = simplifier.simplify()
        print(f"✓ Simplified to {simplified_data.shape[0]} metrics across {simplified_data.shape[1]} periods")
        
        # Show data quality warnings
        quality_report = simplifier.get_data_quality_report()
        if quality_report.get('warnings'):
            print(f"\n⚠️  Data Quality Warnings:")
            for warning in quality_report['warnings']:
                print(f"   - {warning}")
        
        # Step 3: Model Selection
        print(f"\n[3/5] Selecting appropriate models...")
        selector = ModelSelector(simplified_data)
        fit_scores = selector.calculate_fit_scores()
        recommended = selector.get_recommended_models(min_score=0.5)
        
        print(f"✓ Evaluated {len(fit_scores)} models")
        print(f"✓ Recommended {len(recommended)} models")
        
        # Step 4: Print Analysis
        if verbose:
            print(f"\n[4/5] Detailed Model Analysis:")
            selector.print_analysis(detailed=True)
        else:
            print(f"\n[4/5] Model Fit Scores:")
            for model, score in sorted(fit_scores.items(), key=lambda x: x[1], reverse=True):
                status = "✅" if model in recommended else "❌"
                print(f"   {status} {model:20s} {score:.2f}")
        
        # Step 5: Calculate Valuations (only recommended models)
        print(f"\n[5/5] Calculating valuations using recommended models...")
        calculator = ValuationCalculator(simplified_data)
        calculator.calculate_all_valuations(models_to_calculate=recommended)
        
        results = calculator.get_results()
        print(f"✓ Calculated {len(results)} valuations")
        
        if results:
            print(f"\n📊 Valuation Results:")
            for model, value in results.items():
                print(f"   {model:20s} ${value:,.2f}")
            
            avg = calculator.get_average_valuation()
            weighted_avg = calculator.get_average_valuation(weighted=True)
            current = calculator.current_price
            
            if avg:
                print(f"\n   {'Average':20s} ${avg:,.2f}")
            if weighted_avg:
                print(f"   {'Weighted Average':20s} ${weighted_avg:,.2f}")
            if current:
                print(f"   {'Current Price':20s} ${current:,.2f}")
                if avg:
                    upside = ((avg - current) / current) * 100
                    print(f"   {'Upside/Downside':20s} {upside:+.1f}%")
        
        return {
            'ticker': ticker,
            'success': True,
            'recommended_models': recommended,
            'fit_scores': fit_scores,
            'valuations': results,
            'num_years': quality_report.get('num_years', 0)
        }
        
    except (DataFetcherError, SimplifierError) as e:
        print(f"\n❌ Error: {e}")
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


def test_multiple_companies(tickers: list, verbose: bool = False):
    """
    Test model selection for multiple companies.
    
    Args:
        tickers: List of stock ticker symbols
        verbose: If True, show detailed analysis for each
    """
    print("\n" + "="*70)
    print("BATCH MODEL SELECTION TEST")
    print("="*70)
    print(f"Testing {len(tickers)} companies: {', '.join(tickers)}")
    
    results = []
    
    for ticker in tickers:
        result = test_single_company(ticker, verbose=verbose)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful: {len(successful)}/{len(tickers)}")
    print(f"❌ Failed: {len(failed)}/{len(tickers)}")
    
    if successful:
        print(f"\n📊 Model Selection Summary:")
        print(f"{'Ticker':<8} {'Years':<6} {'Models':<8} {'Top Models'}")
        print("-" * 70)
        
        for r in successful:
            ticker = r['ticker']
            years = r.get('num_years', 'N/A')
            num_models = len(r.get('recommended_models', []))
            
            # Get top 3 models by score
            fit_scores = r.get('fit_scores', {})
            top_models = sorted(fit_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_names = ', '.join([m[0][:3].upper() for m in top_models])
            
            print(f"{ticker:<8} {years!s:<6} {num_models:<8} {top_names}")
    
    if failed:
        print(f"\n❌ Failed Companies:")
        for r in failed:
            print(f"   {r['ticker']}: {r.get('error', 'Unknown error')}")


def run_comprehensive_test():
    """Run a comprehensive test with various company types."""
    
    test_cases = {
        'Dividend Aristocrats': ['KO', 'JNJ'],
        'Growth Stocks': ['NVDA', 'TSLA'],
        'Financial Institutions': ['BAC', 'JPM'],
        'Cyclical Companies': ['F'],
        'Tech Giants': ['AAPL', 'MSFT'],
    }
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL SELECTOR TEST")
    print("="*70)
    print("\nTesting different company categories to validate model selection logic")
    
    all_results = {}
    
    for category, tickers in test_cases.items():
        print(f"\n\n{'='*70}")
        print(f"CATEGORY: {category}")
        print('='*70)
        
        category_results = []
        for ticker in tickers:
            result = test_single_company(ticker, verbose=False)
            category_results.append(result)
        
        all_results[category] = category_results
    
    # Final summary by category
    print("\n\n" + "="*70)
    print("FINAL SUMMARY BY CATEGORY")
    print("="*70)
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        
        for r in results:
            if r['success']:
                ticker = r['ticker']
                recommended = r.get('recommended_models', [])
                print(f"  {ticker}: {len(recommended)} models - {', '.join(recommended)}")
            else:
                print(f"  {r['ticker']}: FAILED - {r.get('error', 'Unknown')}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--comprehensive':
            # Run comprehensive test
            run_comprehensive_test()
        elif sys.argv[1] == '--batch':
            # Batch test with provided tickers
            tickers = sys.argv[2:] if len(sys.argv) > 2 else ['AAPL', 'MSFT', 'KO', 'NVDA']
            test_multiple_companies(tickers, verbose=False)
        else:
            # Single ticker test (verbose)
            ticker = sys.argv[1].upper()
            test_single_company(ticker, verbose=True)
    else:
        # Default: Test a few representative companies
        print("\n" + "="*70)
        print("MODEL SELECTOR - QUICK TEST")
        print("="*70)
        print("\nUsage:")
        print("  python test_model_selector.py AAPL           # Test single ticker")
        print("  python test_model_selector.py --batch AAPL MSFT KO  # Test multiple")
        print("  python test_model_selector.py --comprehensive  # Full test suite")
        print("\nRunning default test with 3 companies...\n")
        
        default_tickers = ['AAPL', 'KO', 'NVDA']
        test_multiple_companies(default_tickers, verbose=False)
        
        print("\n💡 Tip: Run with --comprehensive for a full test of all company types")