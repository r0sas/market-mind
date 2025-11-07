# transform_columns.py
import pandas as pd

def _transform_columns(df):
    # Convert date-based column headers to a year-only format
    df.columns = pd.to_datetime(df.columns).year
    # Transpose the DataFrame
    df = df.T.reset_index()
    df.columns = ['Year'] + list(df.iloc[0, 1:])
    df = df.drop(0)
    return df
