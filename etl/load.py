"""
load.py
--------
Purpose: Load the cleaned arrivals data AND the Prophet forecast results
into their respective PostgreSQL tables.

This is the final step of the ETL pipeline:
    PDF -> extract.py -> clean.py -> load.py -> PostgreSQL

Usage:
    python etl/load.py
"""

import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now get credentials from environment
DB_USER = os.getenv('DB_USER', 'tourism_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'tourism_pass123')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'tourism_db')

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

ARRIVALS_CSV = 'data/processed/arrivals_clean.csv'
FORECAST_CSV = 'data/processed/forecast_future.csv'


def load_arrivals(engine):
    df = pd.read_csv(ARRIVALS_CSV, parse_dates=['date'])

    with engine.begin() as conn:
        # Clear existing rows first rather than using pandas' if_exists='replace',
        # which would DROP the table and lose our constraints/indexes from schema.sql.
        conn.execute(text("DELETE FROM tourist_arrivals"))

    df[['date', 'arrivals']].to_sql(
        'tourist_arrivals', engine, if_exists='append', index=False
    )
    print(f"Loaded {len(df)} rows into tourist_arrivals")


def load_forecast(engine):
    df = pd.read_csv(FORECAST_CSV, parse_dates=['month'])
    df['model_name'] = 'prophet'

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM forecast_results WHERE model_name = 'prophet'"))

    df[['month', 'predicted_arrivals', 'lower_bound', 'upper_bound', 'model_name']].to_sql(
        'forecast_results', engine, if_exists='append', index=False
    )
    print(f"Loaded {len(df)} rows into forecast_results")


def verify(engine):
    with engine.connect() as conn:
        arrivals_count = conn.execute(text("SELECT COUNT(*) FROM tourist_arrivals")).scalar()
        forecast_count = conn.execute(text("SELECT COUNT(*) FROM forecast_results")).scalar()
        date_range = conn.execute(text(
            "SELECT MIN(date), MAX(date) FROM tourist_arrivals"
        )).fetchone()

    print(f"\nVerification:")
    print(f"   tourist_arrivals: {arrivals_count} rows, {date_range[0]} to {date_range[1]}")
    print(f"   forecast_results: {forecast_count} rows")


if __name__ == "__main__":
    try:
        engine = create_engine(CONNECTION_STRING)
        
        # Test connection
        with engine.connect() as conn:
            print("✅ Connected to PostgreSQL")
        
        load_arrivals(engine)
        load_forecast(engine)
        verify(engine)

        print("\n✅ Done. Data is now in PostgreSQL.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check credentials in .env file")
        print("   3. Create database first: psql -U postgres -c 'CREATE DATABASE tourism_db;'")
        print("   4. Create user: psql -U postgres -c \"CREATE USER tourism_user WITH PASSWORD 'tourism_pass123';\"")