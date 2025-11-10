import pandas as pd
import logging
from core.simplifier.simplifier_exceptions import SimplifierError

logger = logging.getLogger(__name__)

class ColumnTransformer:
    """Handles transformation of columns to year-based format."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transform_columns_to_years(self) -> pd.DataFrame:
        """Convert columns to 'Year N' format and add a Date row with actual years."""
        try:
            years = []
            for col in self.df.columns:
                try:
                    year = pd.to_datetime(col).year
                    years.append(year)
                except Exception as e:
                    logger.warning(f"Could not parse date from column '{col}': {e}")
                    years.append(None)

            if all(y is None for y in years):
                raise SimplifierError("Could not parse any valid dates from column names")

            self.df.loc["Date"] = years

            num_cols = len(self.df.columns)
            new_col_names = [f"Year {num_cols - i}" for i in range(num_cols)]
            self.df.columns = new_col_names

            # Move Date row to top
            other_rows = [idx for idx in self.df.index if idx != "Date"]
            self.df = self.df.loc[["Date"] + other_rows]

            logger.info(f"Transformed {num_cols} columns to year-based format.")
            return self.df

        except Exception as e:
            raise SimplifierError(f"Failed to transform columns to years: {str(e)}")
