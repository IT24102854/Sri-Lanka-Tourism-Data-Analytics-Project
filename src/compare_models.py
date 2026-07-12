"""
src/compare_models.py
----------------------
Purpose: Compare three forecasting approaches on the SAME train/test
split (train on 2018-2024, test on 2025) so we can honestly say which
model performs best on this dataset, backed by real numbers.

Models compared:
  1. Linear Regression  - simplest baseline (trend + calendar month as features)
  2. SARIMA             - classical statistical time series model
  3. Prophet             - the model we already tuned and use in production

Usage:
    python src/compare_models.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

TEST_MONTHS = 12


def load_data():
    """Load the cleaned data"""
    df = pd.read_csv('data/processed/arrivals_clean.csv', parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)


def evaluate(actual, predicted, model_name):
    """Calculate accuracy metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    # Additional metrics
    mpe = np.mean((actual - predicted) / actual) * 100  # Mean Percentage Error
    r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2))
    
    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mpe': mpe,
        'r2': r2
    }


def run_linear_regression(df, train, test):
    """
    Simple baseline: represents time as a plain incrementing index (t),
    plus the calendar month as a category, so the model can pick up a
    linear trend AND a repeating seasonal pattern -- without any of
    Prophet/SARIMA's more sophisticated trend-bending logic.
    """
    df = df.copy()
    df['t'] = range(len(df))
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add year as feature (helps with trend)
    df['year'] = df['date'].dt.year
    df['year_scaled'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    features = ['t', 'month_sin', 'month_cos', 'year_scaled']
    X = df[features]
    y = df['arrivals']

    X_train, X_test = X.iloc[:-TEST_MONTHS], X.iloc[-TEST_MONTHS:]
    y_train, y_test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, None)  # arrivals can't be negative

    return evaluate(y_test, predictions, 'Linear Regression'), predictions, model


def run_sarima(train, test):
    """
    SARIMAX(1,1,1)(1,1,1,12): a standard starting configuration for
    monthly data with yearly seasonality (the '12' is the seasonal period).
    """
    try:
        model = SARIMAX(
            train['arrivals'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=TEST_MONTHS)
        predictions = np.clip(forecast.values, 0, None)
        return evaluate(test['arrivals'].values, predictions, 'SARIMA'), predictions, fitted
    except Exception as e:
        print(f"⚠️ SARIMA failed: {e}")
        # Fallback: return NaN values
        predictions = np.full(TEST_MONTHS, np.nan)
        return {
            'model': 'SARIMA',
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'mpe': np.nan,
            'r2': np.nan
        }, predictions, None


def run_prophet(train, test):
    """Same tuned settings established earlier in this project."""
    prophet_train = train.rename(columns={'date': 'ds', 'arrivals': 'y'})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        changepoint_range=0.95
    )
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=TEST_MONTHS, freq='MS')
    forecast = model.predict(future)
    predictions = forecast[forecast['ds'].isin(test['date'])]['yhat'].values
    predictions = np.clip(predictions, 0, None)

    return evaluate(test['arrivals'].values, predictions, 'Prophet'), predictions, model


def create_comparison_visualization(df, test, results, all_predictions):
    """Create comparison charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Actual vs Predictions
    ax1 = axes[0, 0]
    months = test['date'].dt.strftime('%b %Y')
    ax1.plot(months, test['arrivals'].values, 'b-o', linewidth=2, markersize=8, label='Actual')
    
    colors = {'Linear Regression': 'green', 'SARIMA': 'orange', 'Prophet': 'red'}
    for model_name, preds in all_predictions.items():
        if preds is not None and not np.isnan(preds).all():
            ax1.plot(months, preds, '--s', linewidth=2, markersize=6, 
                    label=model_name, color=colors.get(model_name, 'gray'))
    
    ax1.set_title('Model Comparison: Actual vs Predictions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Arrivals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # 2. Accuracy Metrics (bar chart)
    ax2 = axes[0, 1]
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna()
    
    if not results_df.empty:
        x = np.arange(len(results_df))
        width = 0.25
        bars1 = ax2.bar(x - width, results_df['mae'], width, label='MAE', color='#3498db')
        bars2 = ax2.bar(x, results_df['rmse'], width, label='RMSE', color='#e74c3c')
        bars3 = ax2.bar(x + width, results_df['mape'] * 100, width, label='MAPE (x100)', color='#2ecc71')
        
        ax2.set_title('Accuracy Metrics Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Error')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['model'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Error Distribution
    ax3 = axes[1, 0]
    for model_name, preds in all_predictions.items():
        if preds is not None and not np.isnan(preds).all():
            errors = preds - test['arrivals'].values
            ax3.hist(errors, bins=10, alpha=0.5, label=model_name)
    
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Error (Predicted - Actual)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. MAPE Bar Chart
    ax4 = axes[1, 1]
    if not results_df.empty:
        colors_bar = ['#2ecc71' if m < 10 else '#f39c12' if m < 15 else '#e74c3c' 
                      for m in results_df['mape']]
        bars = ax4.bar(results_df['model'], results_df['mape'], color=colors_bar, edgecolor='black')
        ax4.set_title('MAPE by Model', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('MAPE (%)')
        ax4.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Excellent (<10%)')
        ax4.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Good (<15%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, results_df['mape']):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/processed/model_comparison.png', dpi=200, bbox_inches='tight')
    print("✅ Saved: data/processed/model_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔄 MODEL COMPARISON: Linear Regression vs SARIMA vs Prophet")
    print("="*70)
    
    # Load data
    df = load_data()
    train = df.iloc[:-TEST_MONTHS].reset_index(drop=True)
    test = df.iloc[-TEST_MONTHS:].reset_index(drop=True)
    
    print(f"\n📊 Data split:")
    print(f"   Train: {train['date'].min().strftime('%Y-%m')} to {train['date'].max().strftime('%Y-%m')} ({len(train)} months)")
    print(f"   Test:  {test['date'].min().strftime('%Y-%m')} to {test['date'].max().strftime('%Y-%m')} ({len(test)} months)")
    
    # Run models
    print("\n🔄 Running models...")
    
    # Linear Regression
    print("   • Linear Regression...")
    lr_result, lr_preds, lr_model = run_linear_regression(df, train, test)
    
    # SARIMA
    print("   • SARIMA...")
    sarima_result, sarima_preds, sarima_model = run_sarima(train, test)
    
    # Prophet
    print("   • Prophet...")
    prophet_result, prophet_preds, prophet_model = run_prophet(train, test)
    
    # Collect results
    results = [lr_result, sarima_result, prophet_result]
    all_predictions = {
        'Linear Regression': lr_preds,
        'SARIMA': sarima_preds,
        'Prophet': prophet_preds
    }
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=['mape'])
    results_df = results_df.sort_values('mape')
    results_df[['mae', 'rmse']] = results_df[['mae', 'rmse']].round(0)
    results_df[['mape', 'mpe']] = results_df[['mape', 'mpe']].round(1)
    results_df['r2'] = results_df['r2'].round(3)
    
    # Print results
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Determine best model
    best_model = results_df.iloc[0]
    print("\n" + "="*70)
    print("🏆 WINNER:")
    print(f"   {best_model['model']} with MAPE = {best_model['mape']:.1f}%")
    print("="*70)
    
    # Save results
    results_df.to_csv('data/processed/model_comparison.csv', index=False)
    print("\n💾 Saved: data/processed/model_comparison.csv")
    
    # Create visualizations
    print("\n📊 Creating comparison visualizations...")
    create_comparison_visualization(df, test, results, all_predictions)
    
    print("\n✅ COMPARISON COMPLETE!")