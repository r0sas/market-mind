# handle_missing_values.py

def _handle_missing_values(df):
    # Remove any row with missing values
    df = df.dropna()
    return df
