-- schema.sql
-- ----------
-- Purpose: Define the two tables our project needs:
--   1. tourist_arrivals  -> the real, historical monthly data (from ETL)
--   2. forecast_results  -> what Prophet predicted, saved so the
--      dashboard can show forecasts without re-running the model
--
-- Run this once, right after creating the database:
--   psql -U tourism_user -d tourism_db -f sql/schema.sql

-- Drop tables first so this script is safely re-runnable during development.
-- (Remove these two lines once your project is "final" so you don't
-- accidentally wipe real data.)
DROP TABLE IF EXISTS forecast_results;
DROP TABLE IF EXISTS tourist_arrivals;

CREATE TABLE tourist_arrivals (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,      -- UNIQUE stops accidental duplicate months
    arrivals INTEGER NOT NULL CHECK (arrivals >= 0),
    loaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE forecast_results (
    id SERIAL PRIMARY KEY,
    month DATE NOT NULL,
    predicted_arrivals INTEGER NOT NULL,
    lower_bound INTEGER NOT NULL,
    upper_bound INTEGER NOT NULL,
    model_name VARCHAR(50) NOT NULL DEFAULT 'prophet',
    generated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (month, model_name)       -- allows re-running the model later
                                       -- and storing a fresh forecast per month
                                       -- without duplicate rows
);

-- A helpful index for the dashboard's most common query pattern:
-- "give me arrivals ordered by date"
CREATE INDEX idx_arrivals_date ON tourist_arrivals (date);
CREATE INDEX idx_forecast_month ON forecast_results (month);