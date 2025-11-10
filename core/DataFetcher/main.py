from core.datafetcher.financial_data_processor import FinancialDataProcessor
from core.datafetcher.data_displayer import DataDisplayer


def main():
    ticker = input("Enter ticker symbol: ").upper().strip()
    
    if not ticker:
        print("Error: No ticker symbol provided")
        return
    
    processor = FinancialDataProcessor(ticker)
    
    try:
        # Get comprehensive data
        combined_data = processor.get_comprehensive_data()
        
        # Display overview
        summary = processor.get_summary()
        DataDisplayer.display_data_overview(
            processor.ticker_symbol,
            summary,
            combined_data
        )
        
        # Optionally save to CSV
        # combined_data.to_csv(f"{ticker}_financial_data.csv")
        
    except Exception as e:
        print(f"Failed to process data for {ticker}: {e}")


if __name__ == "__main__":
    main()