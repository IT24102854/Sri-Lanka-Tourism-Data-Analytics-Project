"""
clean.py
---------
Purpose: Combine ALL the raw CSVs produced by extract.py into a single, 
clean, deduplicated monthly time series.

This version drops 2017 (incomplete data) and only keeps 2018-2025.

Usage:
    python etl/clean.py data/raw/dec*.csv data/processed/arrivals_clean.csv
"""

import sys
import glob
import pandas as pd

MONTH_NUMBER = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Define the years we want to keep (2017 excluded)
KEEP_YEARS = list(range(2018, 2026))  # 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025


def load_all_raw_csvs(file_paths):
    """Reads every raw CSV and stacks them into one big table."""
    all_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        df['source_file'] = path
        all_frames.append(df)

    if not all_frames:
        raise ValueError("No CSV files found. Check your file path/pattern.")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def clean_and_deduplicate(combined_df):
    """Clean, deduplicate, and filter to only keep 2018-2025."""
    
    # Filter to only keep years 2018-2025
    combined_df = combined_df[combined_df['year'].isin(KEEP_YEARS)]
    
    # Build date column
    combined_df['month_num'] = combined_df['month'].map(MONTH_NUMBER)
    combined_df['date'] = pd.to_datetime(
        dict(year=combined_df['year'], month=combined_df['month_num'], day=1)
    )

    # Deduplicate: keep the latest report version
    combined_df['report_year_hint'] = combined_df['source_file'].str.extract(r'(\d{4})').astype(float)
    combined_df = combined_df.sort_values(['date', 'report_year_hint'])
    deduped = combined_df.drop_duplicates(subset='date', keep='last')

    deduped = deduped[['date', 'arrivals']].sort_values('date').reset_index(drop=True)
    return deduped


def check_for_gaps(df):
    """Check if any months are missing between first and last date."""
    full_range = pd.date_range(df['date'].min(), df['date'].max(), freq='MS')
    missing = set(full_range) - set(df['date'])
    if missing:
        print(f"WARNING: {len(missing)} missing month(s) detected:")
        for m in sorted(missing):
            print("   -", m.strftime('%Y-%m'))
        return missing
    else:
        print("✅ No missing months. Time series is continuous (2018-2025).")
        return set()


def validate_data(df):
    """Perform basic data quality checks."""
    print("\n📊 Data Quality Report:")
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"   Years: {sorted(df['date'].dt.year.unique())}")
    
    # Check for negative arrivals
    negative = df[df['arrivals'] < 0]
    if not negative.empty:
        print(f"   ⚠️ Found {len(negative)} negative arrival values!")
    
    # Check for zero arrivals
    zero = df[df['arrivals'] == 0]
    if not zero.empty:
        print(f"   ℹ️ Found {len(zero)} months with 0 arrivals (COVID-19 period)")
        zero_months = zero['date'].dt.strftime('%Y-%m').tolist()
        print(f"      Months: {zero_months}")
    
    # Summary statistics
    print(f"\n📈 Arrivals Statistics:")
    print(f"   Min: {df['arrivals'].min():,.0f}")
    print(f"   Max: {df['arrivals'].max():,.0f}")
    print(f"   Mean: {df['arrivals'].mean():,.0f}")
    print(f"   Median: {df['arrivals'].median():,.0f}")
    print(f"   Total (2018-2025): {df['arrivals'].sum():,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python etl/clean.py <input_csv_1> [<input_csv_2> ...] <output_csv>")
        sys.exit(1)

    *input_patterns, output_csv = sys.argv[1:]

    # Expand any wildcard patterns
    input_files = []
    for pattern in input_patterns:
        input_files.extend(glob.glob(pattern))
    input_files = sorted(set(input_files))

    print(f"Found {len(input_files)} input file(s):")
    for f in input_files:
        print("   -", f)

    combined = load_all_raw_csvs(input_files)
    print(f"\n📊 Total rows before filtering: {len(combined)}")
    print(f"   Years before filtering: {sorted(combined['year'].unique())}")
    
    clean_df = clean_and_deduplicate(combined)
    print(f"📊 Rows after filtering (2018-2025) and deduplication: {len(clean_df)}")
    print(f"   Years kept: {sorted(clean_df['date'].dt.year.unique())}")
    
    missing = check_for_gaps(clean_df)
    validate_data(clean_df)

    clean_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved {len(clean_df)} clean monthly rows (2018-2025) -> {output_csv}")