"""
model.py
---------
Purpose: Train a Prophet forecasting model on the cleaned tourist
arrivals data, evaluate it against a held-out test period, then
produce a forward-looking forecast using all available data.

Two-phase approach:
  Phase 1: Train on all months EXCEPT the last 12 -> evaluate against
           those last 12 real months -> get honest accuracy numbers.
  Phase 2: Retrain on ALL months -> forecast the NEXT 12 months forward.
           This is the model you'd actually use/deploy.

Usage:
    python src/model.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import logging
import os

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# File paths
INPUT_CSV = 'data/processed/arrivals_clean.csv'
INPUT_CSV_FALLBACK = 'data/processed/arrivals_clean_filled.csv'

TEST_MONTHS = 12
FORECAST_MONTHS = 12

# ============================================
# IMPROVED MODEL PARAMETERS
# ============================================
MODEL_PARAMS = dict(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.8,
    changepoint_range=0.95,
    seasonality_prior_scale=10.0,
    seasonality_mode='additive',
    holidays_prior_scale=2.0
)


def create_holidays():
    """
    Create holidays dataframe for Sri Lanka.
    Each holiday must have the same number of entries.
    """
    # Create a base list of years for consistency
    years = list(range(2018, 2026))  # 2018-2025
    
    # Generate holiday dates (each holiday must have one entry per year)
    holiday_data = []
    
    for year in years:
        # Easter Sunday (varies by year)
        easter_dates = {
            2018: '2018-04-01',
            2019: '2019-04-21',
            2020: '2020-04-12',
            2021: '2021-04-04',
            2022: '2022-04-17',
            2023: '2023-04-09',
            2024: '2024-03-31',
            2025: '2025-04-20'
        }
        holiday_data.append(('Easter_Sunday', easter_dates[year]))
        
        # Christmas (fixed)
        holiday_data.append(('Christmas', f'{year}-12-25'))
        
        # Sinhala & Tamil New Year (around April 13-14)
        new_year_dates = {
            2018: '2018-04-14',
            2019: '2019-04-14',
            2020: '2020-04-14',
            2021: '2021-04-14',
            2022: '2022-04-14',
            2023: '2023-04-14',
            2024: '2024-04-14',
            2025: '2025-04-14'
        }
        holiday_data.append(('Sinhala_Tamil_New_Year', new_year_dates[year]))
        
        # Vesak Full Moon (varies by year)
        vesak_dates = {
            2018: '2018-04-29',
            2019: '2019-05-19',
            2020: '2020-05-07',
            2021: '2021-05-26',
            2022: '2022-05-16',
            2023: '2023-05-06',
            2024: '2024-05-23',
            2025: '2025-05-12'
        }
        holiday_data.append(('Vesak_Full_Moon', vesak_dates[year]))
        
        # Ramadan/Eid (varies by year)
        ramadan_dates = {
            2018: '2018-06-15',
            2019: '2019-06-05',
            2020: '2020-05-24',
            2021: '2021-05-13',
            2022: '2022-05-03',
            2023: '2023-04-23',
            2024: '2024-04-10',
            2025: '2025-03-31'
        }
        holiday_data.append(('Ramadan_Eid', ramadan_dates[year]))
        
        # Deepavali (varies by year)
        deepavali_dates = {
            2018: '2018-11-06',
            2019: '2019-10-27',
            2020: '2020-11-14',
            2021: '2021-11-04',
            2022: '2022-10-24',
            2023: '2023-11-12',
            2024: '2024-10-31',
            2025: '2025-10-20'
        }
        holiday_data.append(('Deepavali', deepavali_dates[year]))
    
    # Create DataFrame
    holidays = pd.DataFrame(holiday_data, columns=['holiday', 'ds'])
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    holidays['lower_window'] = -3
    holidays['upper_window'] = 3
    
    return holidays


def load_data():
    """Load data, trying multiple file paths"""
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
        print(f"✅ Loaded: {INPUT_CSV}")
    elif os.path.exists(INPUT_CSV_FALLBACK):
        df = pd.read_csv(INPUT_CSV_FALLBACK, parse_dates=['date'])
        print(f"✅ Loaded: {INPUT_CSV_FALLBACK}")
    else:
        raise FileNotFoundError(f"Could not find {INPUT_CSV} or {INPUT_CSV_FALLBACK}")
    
    df = df.sort_values('date').reset_index(drop=True)
    return df.rename(columns={'date': 'ds', 'arrivals': 'y'})


def create_model():
    """Create Prophet model with parameters and holidays"""
    holidays = create_holidays()
    
    model = Prophet(
        **MODEL_PARAMS,
        holidays=holidays
    )
    
    # Add custom monthly seasonality
    model.add_seasonality(
        name='monthly',
        period=12,
        fourier_order=5
    )
    
    return model


def evaluate_on_holdout(df):
    """Evaluate model on test period"""
    train = df.iloc[:-TEST_MONTHS].reset_index(drop=True)
    test = df.iloc[-TEST_MONTHS:].reset_index(drop=True)

    print(f"\n📊 Training on: {train['ds'].min().strftime('%Y-%m')} to {train['ds'].max().strftime('%Y-%m')} ({len(train)} months)")
    print(f"📊 Testing on: {test['ds'].min().strftime('%Y-%m')} to {test['ds'].max().strftime('%Y-%m')} ({len(test)} months)")

    model = create_model()
    model.fit(train)

    future = model.make_future_dataframe(periods=TEST_MONTHS, freq='MS')
    forecast = model.predict(future)

    forecast_test = forecast[forecast['ds'].isin(test['ds'])][['ds', 'yhat']]
    comparison = test.merge(forecast_test, on='ds')

    mae = mean_absolute_error(comparison['y'], comparison['yhat'])
    rmse = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))
    mape = mean_absolute_percentage_error(comparison['y'], comparison['yhat']) * 100

    print(f"\n{'='*50}")
    print("HOLDOUT EVALUATION")
    print("="*50)
    print(f"  MAE:  {mae:>10,.0f}  (average error in tourists)")
    print(f"  RMSE: {rmse:>10,.0f}  (root mean squared error)")
    print(f"  MAPE: {mape:>10.1f}% (average percentage error)")
    print("="*50)

    # Comparison with previous model
    print(f"\n📊 Comparison with previous model:")
    print(f"   Previous MAPE: 11.9%")
    print(f"   Current MAPE:  {mape:.1f}%")
    improvement = ((11.9 - mape) / 11.9) * 100 if mape < 11.9 else 0
    if improvement > 0:
        print(f"   ✅ IMPROVEMENT: {improvement:.1f}% better")
    else:
        print(f"   ⚠️ No improvement yet (try tuning further)")

    # Interpretation
    print(f"\n💡 Quality Assessment:")
    if mape < 8:
        print("   ⭐ EXCELLENT: MAPE < 8% - Very accurate model")
    elif mape < 10:
        print("   ✅ GOOD: MAPE < 10% - Accurate model")
    elif mape < 15:
        print("   👍 DECENT: MAPE < 15% - Reasonably accurate")
    else:
        print("   ⚠️ NEEDS IMPROVEMENT: MAPE > 15% - Consider more tuning")

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def train_final_and_forecast(df):
    """Retrain on all data and forecast 2026"""
    print(f"\n🔄 Training final model on all {len(df)} months...")
    
    model = create_model()
    model.fit(df)

    future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq='MS')
    forecast = model.predict(future)

    # Extract only 2026 forecast
    future_only = forecast[forecast['ds'] > df['ds'].max()][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ]
    future_only.columns = ['month', 'predicted_arrivals', 'lower_bound', 'upper_bound']
    future_only['month'] = future_only['month'].dt.strftime('%Y-%m')
    
    # Round numeric columns
    future_only['predicted_arrivals'] = future_only['predicted_arrivals'].round(0)
    future_only['lower_bound'] = future_only['lower_bound'].round(0)
    future_only['upper_bound'] = future_only['upper_bound'].round(0)

    return model, future_only


def print_forecast_summary(forecast_df, df):
    """Print formatted forecast summary"""
    print("\n" + "="*50)
    print("2026 FORECAST SUMMARY")
    print("="*50)
    print(forecast_df.to_string(index=False))
    
    # Calculate totals
    total_2025 = df[df['ds'].dt.year == 2025]['y'].sum()
    total_2026 = forecast_df['predicted_arrivals'].sum()
    growth = ((total_2026 - total_2025) / total_2025) * 100
    
    print(f"\n📈 2025 Actual: {total_2025:>10,.0f}")
    print(f"📈 2026 Forecast: {total_2026:>10,.0f}")
    print(f"📈 Growth: {growth:>10.1f}%")
    
    # Seasonal highlights
    max_month = forecast_df.loc[forecast_df['predicted_arrivals'].idxmax()]
    min_month = forecast_df.loc[forecast_df['predicted_arrivals'].idxmin()]
    print(f"\n🏆 Peak Month: {max_month['month']} ({max_month['predicted_arrivals']:,.0f} arrivals)")
    print(f"📉 Low Month: {min_month['month']} ({min_month['predicted_arrivals']:,.0f} arrivals)")


def visualize_forecast(df, forecast_df, model):
    """Create professional forecast visualization"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Get full forecast for visualization
        future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq='MS')
        full_forecast = model.predict(future)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Chart 1: Historical + Forecast
        ax1 = axes[0]
        ax1.plot(df['ds'], df['y'], 'b-', linewidth=2, label='Historical')
        ax1.plot(full_forecast['ds'], full_forecast['yhat'], 'r-', linewidth=2, label='Forecast')
        ax1.fill_between(full_forecast['ds'], 
                        full_forecast['yhat_lower'], 
                        full_forecast['yhat_upper'],
                        color='red', alpha=0.2, label='80% Confidence Interval')
        
        # Mark forecast start
        forecast_start = df['ds'].max()
        ax1.axvline(x=forecast_start, color='green', linestyle='--', alpha=0.7)
        ax1.text(forecast_start, ax1.get_ylim()[1]*0.95, 'Forecast Start', 
                 color='green', ha='right', fontsize=11)
        
        ax1.set_title('Sri Lanka Tourism: Historical + 2026 Forecast', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Monthly Arrivals')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        
        # Chart 2: 2026 Only
        ax2 = axes[1]
        months = forecast_df['month'].str[:3]  # Jan, Feb, ...
        bars = ax2.bar(months, forecast_df['predicted_arrivals'], 
                      color='#2ecc71', edgecolor='black', alpha=0.7)
        ax2.errorbar(months, forecast_df['predicted_arrivals'],
                    yerr=[forecast_df['predicted_arrivals'] - forecast_df['lower_bound'],
                          forecast_df['upper_bound'] - forecast_df['predicted_arrivals']],
                    fmt='none', color='red', capsize=5, label='80% CI')
        
        # Add value labels
        for bar, val in zip(bars, forecast_df['predicted_arrivals']):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 3000,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax2.set_title('2026 Monthly Forecast', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Forecasted Arrivals')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        
        plt.tight_layout()
        plt.savefig('data/processed/forecast_2026_improved.png', dpi=150, bbox_inches='tight')
        print("\n✅ Saved: data/processed/forecast_2026_improved.png")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")


if __name__ == "__main__":
    try:
        df = load_data()
        
        print(f"\n📊 Data loaded: {len(df)} rows")
        print(f"📅 Date range: {df['ds'].min().strftime('%Y-%m')} to {df['ds'].max().strftime('%Y-%m')}")
        print(f"📅 Years: {sorted(df['ds'].dt.year.unique())}")
        
        # Evaluate on holdout
        metrics = evaluate_on_holdout(df)

        # Final forecast
        final_model, future_forecast = train_final_and_forecast(df)
        print_forecast_summary(future_forecast, df)

        # Save results
        future_forecast.to_csv('data/processed/forecast_future.csv', index=False)
        with open('data/processed/prophet_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)

        print("\n✅ Saved: data/processed/forecast_future.csv")
        print("✅ Saved: data/processed/prophet_model.pkl")
        
        # Create visualization
        visualize_forecast(df, future_forecast, final_model)
        
        print("\n" + "="*50)
        print("✅ IMPROVED MODEL COMPLETE!")
        print("="*50)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()