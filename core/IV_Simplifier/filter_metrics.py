# filter_metrics.py

def _filter_metrics(df):
    # Retain only predefined essential financial metrics
    essential_metrics = ['Free Cash Flow', 'EPS', 'P/E Ratio']
    df = df[df['Year'].isin(essential_metrics)]
    return df
