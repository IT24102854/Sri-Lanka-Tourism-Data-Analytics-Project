"""
eda.py
------
Purpose: Exploratory Data Analysis on the cleaned Sri Lanka tourist
arrivals dataset (2018-2025). Produces 5 charts that tell the story of
the data: overall trend with shock events, seasonality, year-over-year
comparison, yearly totals, and the smoothed long-term trend.

This is meant to be run as a script, or copy-pasted cell-by-cell into
a Jupyter notebook (notebooks/eda.ipynb) for your submission.

Usage:
    python notebooks/eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------
# Step 1: Load and inspect
# ---------------------------------------------------------------
df = pd.read_csv('data/processed/arrivals_clean.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print("Shape:", df.shape)
print("Date range:", df['date'].min().date(), "to", df['date'].max().date())
print(df['arrivals'].describe())

# ---------------------------------------------------------------
# Step 2: Time series with shock events marked
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df['date'], df['arrivals'], color='#1f77b4', linewidth=1.8)

events = [
    ('2019-04-01', 'Easter Sunday\nattacks'),
    ('2020-03-01', 'COVID-19\nborder closure'),
    ('2022-03-01', 'Economic crisis\n(fuel/forex shortage)'),
]
for date_str, label in events:
    event_date = pd.to_datetime(date_str)
    ax.axvline(event_date, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(event_date, ax.get_ylim()[1]*0.95, label, fontsize=9, color='red', ha='left', va='top')

ax.set_title('Sri Lanka Monthly Tourist Arrivals, 2018–2025', fontsize=14, fontweight='bold')
ax.set_ylabel('Tourist Arrivals')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('eda_01_timeseries_events.png', dpi=130)
plt.close()

# ---------------------------------------------------------------
# Step 3: Seasonality boxplot (excluding COVID years)
# ---------------------------------------------------------------
df['month_name'] = df['date'].dt.strftime('%b')
df['month_num'] = df['date'].dt.month
normal_years = df[~df['date'].dt.year.isin([2020, 2021])]
order = normal_years.sort_values('month_num')['month_name'].unique()

fig, ax = plt.subplots(figsize=(12, 6))
normal_years.boxplot(column='arrivals', by='month_name', ax=ax, positions=range(len(order)))
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_title('Seasonality: Arrivals by Month (excluding COVID years 2020-2021)', fontsize=13, fontweight='bold')
ax.set_ylabel('Tourist Arrivals')
plt.suptitle('')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eda_02_seasonality_boxplot.png', dpi=130)
plt.close()

# ---------------------------------------------------------------
# Step 4: Year-over-year overlay
# ---------------------------------------------------------------
df['year'] = df['date'].dt.year
years = sorted(df['year'].unique())
colors = plt.cm.viridis([i / (len(years)-1) for i in range(len(years))])

fig, ax = plt.subplots(figsize=(12, 7))
for year, color in zip(years, colors):
    year_data = df[df['year'] == year].sort_values('month_num')
    linewidth = 2.5 if year in (2018, 2025) else 1.3
    ax.plot(year_data['month_num'], year_data['arrivals'], label=str(year), color=color, linewidth=linewidth)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Year-over-Year Comparison: Monthly Arrivals by Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Tourist Arrivals')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(title='Year', loc='upper left', ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eda_03_year_over_year.png', dpi=130)
plt.close()

# ---------------------------------------------------------------
# Step 5: Yearly totals bar chart
# ---------------------------------------------------------------
yearly = df.groupby('year')['arrivals'].sum()

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(yearly.index.astype(str), yearly.values, color='#2a6f97')
for bar, year in zip(bars, yearly.index):
    if year in (2020, 2021):
        bar.set_color('#c0392b')
for bar, value in zip(bars, yearly.values):
    ax.text(bar.get_x() + bar.get_width()/2, value + 20000, f'{value:,}', ha='center', fontsize=9)

ax.set_title('Total Annual Tourist Arrivals, 2018–2025', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Arrivals')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('eda_04_yearly_totals.png', dpi=130)
plt.close()

print("\nYearly totals:")
print(yearly)
print("\nYear-over-year % change:")
print(yearly.pct_change().mul(100).round(1))

# ---------------------------------------------------------------
# Step 6: 12-month rolling average trend
# ---------------------------------------------------------------
df['rolling_12m'] = df['arrivals'].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['date'], df['arrivals'], color='lightgray', linewidth=1, label='Actual monthly arrivals')
ax.plot(df['date'], df['rolling_12m'], color='#d62828', linewidth=2.5, label='12-month rolling average (trend)')
ax.set_title('Underlying Trend: 12-Month Rolling Average', fontsize=13, fontweight='bold')
ax.set_ylabel('Tourist Arrivals')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eda_05_rolling_trend.png', dpi=130)
plt.close()

print("\nAll 5 charts saved.")