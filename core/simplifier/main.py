# main.py
import logging
from core.datafetcher.data_fetcher import DataFetcher, DataFetcherError
from core.datafetcher.financial_data_processor import FinancialDataProcessor, FinancialDataProcessorError
from core.simplifier.iv_simplifier import IVSimplifier
from core.simplifier.simplifier_exceptions import SimplifierError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(ticker: str) -> None:
    """
    Entry point for retrieving, processing, and simplifying financial data
    for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT')
    """
    logger.info(f"Starting pipeline for ticker: {ticker}")

    try:
        # Step 1: Fetch raw data
        fetcher = DataFetcher(ticker)
        raw_data = fetcher.get_comprehensive_data()
        logger.info(f"Fetched raw data for {ticker} (shape={raw_data.shape})")

        # Step 2: Process data
        processor = FinancialDataProcessor(raw_data)
        processed_data = processor.process_all()
        logger.info(f"Processed data for {ticker} (shape={processed_data.shape})")

        # Step 3: Simplify for intrinsic value calculation
        simplifier = IVSimplifier(processed_data)
        simplified_data = simplifier.simplify()
        logger.info(f"Simplified data for {ticker} (shape={simplified_data.shape})")

        # Step 4: Display summary or export
        print(f"\n{'='*60}")
        print(f"INTRINSIC VALUE DATA FOR {ticker}")
        print(f"{'='*60}\n")
        print(simplified_data.head(10))

    except (DataFetcherError, FinancialDataProcessorError, SimplifierError) as e:
        logger.error(f"Pipeline failed for {ticker}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error for {ticker}: {e}")


if __name__ == "__main__":
    # Example usage
    main("MSFT")
