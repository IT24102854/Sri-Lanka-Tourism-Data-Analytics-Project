"""
visualize_forecast.py
--------------------
Create professional visualizations of the forecast results.

Usage:
    python src/visualize_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style - using available matplotlib styles
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# ============================================
# LOAD DATA
# ============================================
print("="*70)
print("FORECAST VISUALIZATION")
print("="*70)

# Load historical data
df = pd.read_csv('data/processed/arrivals_clean_filled.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Load forecast
forecast_df = pd.read_csv('data/processed/forecast_future.csv')
forecast_df['month'] = pd.to_datetime(forecast_df['month'])

# Load the model
with open('data/processed/prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"\n📊 Historical data: {len(df)} rows")
print(f"📊 Forecast: {len(forecast_df)} months (2026)")

# Get full forecast for plotting
future = model.make_future_dataframe(periods=12, freq='MS')
full_forecast = model.predict(future)

# Calculate key metrics
total_2026 = forecast_df['predicted_arrivals'].sum()
total_2025 = df[df['date'].dt.year == 2025]['arrivals'].sum()
growth = ((total_2026 - total_2025) / total_2025) * 100
peak_month = forecast_df.loc[forecast_df['predicted_arrivals'].idxmax()]
low_month = forecast_df.loc[forecast_df['predicted_arrivals'].idxmin()]

# ============================================
# FIGURE 1: COMPLETE TIME SERIES WITH FORECAST
# ============================================
print("\n📊 Creating Figure 1: Complete Time Series...")

fig, ax = plt.subplots(figsize=(16, 8))

# Historical data
ax.plot(df['date'], df['arrivals'], 'b-', linewidth=2.5, alpha=0.8, label='Historical Data')

# Forecast
forecast_mask = full_forecast['ds'] > df['date'].max()
ax.plot(full_forecast[forecast_mask]['ds'], 
        full_forecast[forecast_mask]['yhat'], 
        'r-', linewidth=2.5, label='Forecast')

# Confidence interval
ax.fill_between(full_forecast[forecast_mask]['ds'],
                full_forecast[forecast_mask]['yhat_lower'],
                full_forecast[forecast_mask]['yhat_upper'],
                color='red', alpha=0.2, label='80% Confidence Interval')

# Add vertical line at forecast start
forecast_start = df['date'].max()
ax.axvline(x=forecast_start, color='green', linestyle='--', alpha=0.7, linewidth=2)
ax.text(forecast_start, ax.get_ylim()[1]*0.95, 'Forecast Start', 
        color='green', ha='right', fontsize=12, fontweight='bold')

# Add event annotations
events = [
    ('2019-04-01', 'Easter\nAttacks', '#e74c3c'),
    ('2020-03-01', 'COVID-19\nLockdown', '#c0392b'),
    ('2022-03-01', 'Economic\nCrisis', '#e67e22'),
]

for date_str, label, color in events:
    event_date = pd.to_datetime(date_str)
    ax.axvline(x=event_date, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(event_date, ax.get_ylim()[1]*0.85, label, 
            fontsize=9, color=color, ha='center', va='top')

# Formatting
ax.set_title('Sri Lanka Tourism: Historical Performance & 2026 Forecast', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Monthly Tourist Arrivals', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Annotate 2026 total
ax.text(pd.to_datetime('2026-06-01'), ax.get_ylim()[1]*0.1,
        f'2026 Total: {total_2026:,.0f} arrivals\n+{growth:.1f}% vs 2025',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('data/processed/fig_forecast_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: data/processed/fig_forecast_timeseries.png")

# ============================================
# FIGURE 2: 2026 FORECAST - MONTHLY DETAIL
# ============================================
print("\n📊 Creating Figure 2: 2026 Monthly Forecast...")

fig, ax = plt.subplots(figsize=(14, 7))

months = forecast_df['month'].dt.strftime('%B')
bars = ax.bar(months, forecast_df['predicted_arrivals'], 
              color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)

# Add error bars for confidence intervals
ax.errorbar(months, forecast_df['predicted_arrivals'],
            yerr=[forecast_df['predicted_arrivals'] - forecast_df['lower_bound'],
                  forecast_df['upper_bound'] - forecast_df['predicted_arrivals']],
            fmt='none', color='red', capsize=8, linewidth=2, label='80% Confidence Interval')

# Add value labels on bars
for bar, val in zip(bars, forecast_df['predicted_arrivals']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 5000,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight peak and low bars
peak_idx = forecast_df[forecast_df['month'] == peak_month['month']].index[0]
low_idx = forecast_df[forecast_df['month'] == low_month['month']].index[0]
bars[peak_idx].set_color('#e74c3c')
bars[peak_idx].set_edgecolor('darkred')
bars[low_idx].set_color('#3498db')
bars[low_idx].set_edgecolor('darkblue')

# Add peak/low annotations
ax.text(peak_idx, peak_month['predicted_arrivals'] + 15000,
        '🏆 PEAK', ha='center', fontsize=11, fontweight='bold', color='darkred')
ax.text(low_idx, low_month['predicted_arrivals'] - 20000,
        '📉 LOW', ha='center', fontsize=11, fontweight='bold', color='darkblue')

ax.set_title('2026 Monthly Forecast with Confidence Intervals', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Forecasted Tourist Arrivals', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/processed/fig_forecast_monthly.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: data/processed/fig_forecast_monthly.png")

# ============================================
# FIGURE 3: YEAR-OVER-YEAR COMPARISON
# ============================================
print("\n📊 Creating Figure 3: Year-over-Year Comparison...")

fig, ax = plt.subplots(figsize=(14, 7))

# Get last 3 years of historical data
years_to_show = [2023, 2024, 2025]
colors_hist = ['#3498db', '#2ecc71', '#f39c12']

for year, color in zip(years_to_show, colors_hist):
    year_data = df[df['date'].dt.year == year].copy()
    year_data['month_num'] = year_data['date'].dt.month
    ax.plot(year_data['month_num'], year_data['arrivals'], 
            'o-', color=color, linewidth=2.5, markersize=8,
            label=f'{year} (Actual)')

# Add 2026 forecast
forecast_2026 = forecast_df.copy()
forecast_2026['month_num'] = forecast_2026['month'].dt.month
ax.plot(forecast_2026['month_num'], forecast_2026['predicted_arrivals'],
        'r-s', linewidth=3, markersize=10, label='2026 (Forecast)')

# Add confidence interval for 2026
ax.fill_between(forecast_2026['month_num'],
                forecast_2026['lower_bound'],
                forecast_2026['upper_bound'],
                color='red', alpha=0.15, label='80% CI (2026)')

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Year-over-Year Comparison: 2023-2026', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Tourist Arrivals', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Add annotation about growth
ax.text(1, ax.get_ylim()[1]*0.9,
        f'2025→2026 Growth: {growth:.1f}%',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('data/processed/fig_yoy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: data/processed/fig_yoy_comparison.png")

# ============================================
# FIGURE 4: SEASONALITY DECOMPOSITION
# ============================================
print("\n📊 Creating Figure 4: Seasonality Components...")

# Get the Prophet components
fig = model.plot_components(full_forecast)
fig.set_size_inches(14, 10)
fig.suptitle('Forecast Components: Trend, Seasonality, and Holidays', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('data/processed/fig_components.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: data/processed/fig_components.png")

# ============================================
# FIGURE 5: ANNUAL TREND WITH FORECAST
# ============================================
print("\n📊 Creating Figure 5: Annual Trend...")

fig, ax = plt.subplots(figsize=(14, 7))

# Calculate annual totals
annual_actual = df.groupby(df['date'].dt.year)['arrivals'].sum()
annual_forecast = pd.Series({2026: forecast_df['predicted_arrivals'].sum()})

# Combine
years = list(annual_actual.index) + [2026]
values = list(annual_actual.values) + [annual_forecast[2026]]

# Create bar chart with color coding
colors = ['#3498db'] * len(annual_actual) + ['#e74c3c']
bars = ax.bar([str(y) for y in years], values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 20000,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add growth rate annotations
for i in range(1, len(values)):
    growth_rate = ((values[i] - values[i-1]) / values[i-1]) * 100
    color = 'green' if growth_rate > 0 else 'red'
    ax.annotate(f'{growth_rate:+.1f}%', 
                xy=(i, values[i]),
                xytext=(i, values[i] + values[i]*0.05),
                ha='center', fontsize=9, fontweight='bold', color=color)

# Highlight forecast start
ax.axvline(x=len(annual_actual) - 0.5, color='green', linestyle='--', alpha=0.7, linewidth=2)
ax.text(len(annual_actual) - 0.5, ax.get_ylim()[1]*0.95, 'Forecast', 
        color='green', ha='center', fontsize=12, fontweight='bold')

ax.set_title('Annual Tourist Arrivals: Historical & Forecast', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Annual Arrivals', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('data/processed/fig_annual_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: data/processed/fig_annual_trend.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)

print("\n📊 Generated the following figures:")
print(f"   1. fig_forecast_timeseries.png   - Complete time series with forecast")
print(f"   2. fig_forecast_monthly.png      - 2026 monthly forecast details")
print(f"   3. fig_yoy_comparison.png        - Year-over-year comparison (2023-2026)")
print(f"   4. fig_components.png            - Forecast components (trend, seasonality)")
print(f"   5. fig_annual_trend.png          - Annual trend with forecast")

print("\n📈 2026 Forecast Summary:")
print(f"   Total: {total_2026:,.0f} arrivals")
print(f"   Growth: {growth:.1f}% vs 2025")
print(f"   Peak Month: {peak_month['month'].strftime('%B')} ({peak_month['predicted_arrivals']:,.0f})")
print(f"   Low Month: {low_month['month'].strftime('%B')} ({low_month['predicted_arrivals']:,.0f})")

print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE!")
print("="*70)