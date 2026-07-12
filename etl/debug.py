import pandas as pd
import glob
import os

def main():
    # Check the cleaned data
    cleaned_path = os.path.join('data', 'processed', 'arrivals_clean.csv')
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path, parse_dates=['date'])
        print('Unique years in cleaned data:')
        print(sorted(df['date'].dt.year.unique()))
        print('\nMonths per year:')
        for year in sorted(df['date'].dt.year.unique()):
            months = df[df['date'].dt.year == year]['date'].dt.month.unique()
            print(f'{year}: {sorted(months)}')
    else:
        print(f'Cleaned file not found: {cleaned_path}')

    print('\n' + '='*50)
    print('Checking raw files:')
    print('='*50)

    # Check raw files
    files = sorted(glob.glob(os.path.join('data', 'raw', 'dec*.csv')))
    if not files:
        print('No raw files found in data/raw matching dec*.csv')
    for f in files:
        try:
            df_raw = pd.read_csv(f)
            years = sorted(df_raw['year'].unique()) if 'year' in df_raw.columns else []
            print(f'{f}: {len(df_raw)} rows, years: {years}')
        except Exception as e:
            print(f'Failed to read {f}: {e}')

if __name__ == '__main__':
    main()