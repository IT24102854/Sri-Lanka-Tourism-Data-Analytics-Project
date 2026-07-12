"""
fix_missing_2022.py
------------------
Purpose: Fill the 2 missing months (Nov-Dec 2022) using interpolation.

Usage:
    python fix_missing_2022.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the clean data
df = pd.read_csv('data/processed/arrivals_clean.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print("=" * 60)
print("FILLING MISSING 2022 MONTHS")
print("=" * 60)

print(f"Current rows: {len(df)}")
print(f"Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

# Check for missing months
full_range = pd.date_range(df['date'].min(), df['date'].max(), freq='MS')
missing = set(full_range) - set(df['date'])

if missing:
    print(f"\nMissing months to fill: {len(missing)}")
    for m in sorted(missing):
        print(f"  {m.strftime('%Y-%m')}")
    
    # Show context (surrounding months)
    print("\nContext (surrounding months):")
    context_dates = pd.date_range('2022-08-01', '2023-02-01', freq='MS')
    for date in context_dates:
        val = df[df['date'] == date]['arrivals']
        if not val.empty:
            print(f"  {date.strftime('%Y-%m')}: {val.values[0]:,.0f}")
        else:
            print(f"  {date.strftime('%Y-%m')}: MISSING")
    
    # Create complete date range
    df_full = df.set_index('date').reindex(full_range)
    df_full.index.name = 'date'
    df_full = df_full.reset_index()
    
    # Interpolate missing values
    df_full['arrivals'] = df_full['arrivals'].interpolate(method='linear')
    
    # Add derived columns
    df_full['year'] = df_full['date'].dt.year
    df_full['month'] = df_full['date'].dt.month
    df_full['month_name'] = df_full['date'].dt.strftime('%B')
    df_full['quarter'] = df_full['date'].dt.quarter
    
    # Save the filled dataset
    output_path = 'data/processed/arrivals_clean_filled.csv'
    df_full.to_csv(output_path, index=False)
    
    print(f"\n✅ Filled {len(missing)} missing months")
    print(f"   New total rows: {len(df_full)}")
    print(f"   Saved to: {output_path}")
    
    # Show the interpolated values with context
    print("\nInterpolated values with context:")
    context_dates = pd.date_range('2022-08-01', '2023-02-01', freq='MS')
    for date in context_dates:
        val = df_full[df_full['date'] == date]['arrivals'].values[0]
        is_filled = date in missing
        marker = "🟢 FILLED" if is_filled else ""
        print(f"  {date.strftime('%Y-%m')}: {val:,.0f} {marker}")
    
    # Show summary stats
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total arrivals (2018-2025): {df_full['arrivals'].sum():,.0f}")
    print(f"Total arrivals (before fill): {df['arrivals'].sum():,.0f}")
    print(f"Difference: {df_full['arrivals'].sum() - df['arrivals'].sum():,.0f}")
    
    # Visualize the fill
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Before - highlight missing
    ax1 = axes[0]
    ax1.plot(df['date'], df['arrivals'], 'b-', linewidth=2, label='Existing Data')
    # Highlight missing months
    for date in missing:
        ax1.axvline(x=date, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Before Interpolation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Arrivals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # After - show filled values
    ax2 = axes[1]
    ax2.plot(df_full['date'], df_full['arrivals'], 'g-', linewidth=2, label='Complete Data')
    # Highlight filled points
    filled_mask = df_full['date'].isin(missing)
    ax2.scatter(df_full[filled_mask]['date'], df_full[filled_mask]['arrivals'], 
               color='red', s=100, zorder=5, label='Interpolated Months')
    ax2.set_title('After Interpolation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Arrivals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/interpolation_2022.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: data/processed/interpolation_2022.png")
    plt.show()
else:
    print("\n✅ No missing months found!")
    df_full = df

print("\n" + "=" * 60)
print("✅ Done! Ready for load.py")
print("=" * 60)