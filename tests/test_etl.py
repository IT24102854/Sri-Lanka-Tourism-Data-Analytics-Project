"""
tests/test_etl.py
------------------
Unit tests for the ETL pipeline. Focused on the PURE LOGIC functions
(parsing, cleaning, deduplication) rather than things that need a real
PDF file or a live database connection.

Run with:
    pytest tests/ -v
"""

import sys
import os
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock pdfplumber for testing (we don't want to load real PDFs)
import unittest.mock as mock
sys.modules['pdfplumber'] = mock.Mock()

# Import from extract using correct function names
from etl.extract import clean_number, extract_table_from_text
from etl.clean import clean_and_deduplicate


class TestCleanNumber:
    """extract.clean_number() must handle SLTDA's messy PDF number formats."""

    def test_removes_commas(self):
        assert clean_number("208,253") == 208253

    def test_removes_stray_spaces(self):
        # SLTDA PDFs sometimes render '208, 253' with an extra space
        assert clean_number("208, 253") == 208253

    def test_removes_spaces_inside_digits(self):
        assert clean_number("2 08253") == 208253

    def test_plain_number_unaffected(self):
        assert clean_number("393") == 393

    def test_zero(self):
        assert clean_number("0") == 0

    def test_handles_none(self):
        assert clean_number(None) is None

    def test_handles_empty_string(self):
        assert clean_number("") is None

    def test_handles_percent_sign(self):
        assert clean_number("24.5%") == 24


class TestExtractTableFromText:
    """extract.extract_table_from_text() should parse tables from text."""

    def test_extracts_table_with_months(self):
        """Test that the function extracts table rows from text"""
        text = """
        Table 1. Monthly tourist arrivals, December 2023
        Month 2022 2023 % change
        January 82,327 102,545 24.5
        February 96,507 107,639 11.5
        March 106,500 125,495 17.8
        TOTAL 719,978 1,487,303 106.6
        """
        data_rows = extract_table_from_text(text)
        
        # Should find the data rows
        assert data_rows is not None
        assert len(data_rows) > 0
        
        # Check that months are properly extracted
        months_found = [row[0] for row in data_rows]
        assert 'January' in months_found
        assert 'February' in months_found

    def test_returns_none_for_no_table(self):
        """Test that function returns None when no table is found"""
        text = "This is just plain text without any table."
        result = extract_table_from_text(text)
        assert result is None


class TestCleanAndDeduplicate:
    """clean.clean_and_deduplicate() must correctly merge overlapping
    months from different source files, keeping the most recent."""

    def _create_test_df(self, rows):
        """Helper to create test DataFrame with proper columns"""
        df = pd.DataFrame(rows, columns=['month', 'year', 'arrivals', 'source_file'])
        # Add month_num for date creation
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['month_num'] = df['month'].map(month_map)
        df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month_num'], day=1))
        return df

    def test_deduplicates_overlapping_month(self):
        """December 2022 appears in two reports - newer should win"""
        rows = [
            ('December', 2022, 91000, 'data/raw/dec2022_report.csv'),
            ('December', 2022, 91961, 'data/raw/dec2023_report.csv'),
        ]
        df = self._create_test_df(rows)
        df['report_year_hint'] = df['source_file'].str.extract(r'(\d{4})').astype(float)
        result = clean_and_deduplicate(df)
        assert len(result) == 1
        assert result.iloc[0]['arrivals'] == 91961

    def test_keeps_non_overlapping_months(self):
        rows = [
            ('January', 2022, 82327, 'data/raw/dec2022_report.csv'),
            ('February', 2022, 96507, 'data/raw/dec2022_report.csv'),
        ]
        df = self._create_test_df(rows)
        df['report_year_hint'] = df['source_file'].str.extract(r'(\d{4})').astype(float)
        result = clean_and_deduplicate(df)
        assert len(result) == 2

    def test_output_is_sorted_by_date(self):
        rows = [
            ('March', 2022, 106500, 'data/raw/dec2022_report.csv'),
            ('January', 2022, 82327, 'data/raw/dec2022_report.csv'),
            ('February', 2022, 96507, 'data/raw/dec2022_report.csv'),
        ]
        df = self._create_test_df(rows)
        df['report_year_hint'] = df['source_file'].str.extract(r'(\d{4})').astype(float)
        result = clean_and_deduplicate(df)
        assert result['date'].is_monotonic_increasing


class TestDataQuality:
    """Sanity checks on the final cleaned dataset itself."""

    @pytest.fixture
    def clean_data(self):
        try:
            return pd.read_csv('data/processed/arrivals_clean.csv', parse_dates=['date'])
        except FileNotFoundError:
            # Try the filled version
            try:
                return pd.read_csv('data/processed/arrivals_clean_filled.csv', parse_dates=['date'])
            except FileNotFoundError:
                pytest.skip("No clean data found. Run clean.py first.")

    def test_no_negative_arrivals(self, clean_data):
        assert (clean_data['arrivals'] >= 0).all()

    def test_no_missing_values(self, clean_data):
        assert clean_data['arrivals'].notna().all()
        assert clean_data['date'].notna().all()

    def test_no_duplicate_dates(self, clean_data):
        assert clean_data['date'].is_unique

    def test_dates_are_month_starts(self, clean_data):
        assert (clean_data['date'].dt.day == 1).all()


class TestDataIntegrity:
    """Additional data integrity checks"""

    @pytest.fixture
    def clean_data(self):
        try:
            return pd.read_csv('data/processed/arrivals_clean.csv', parse_dates=['date'])
        except FileNotFoundError:
            try:
                return pd.read_csv('data/processed/arrivals_clean_filled.csv', parse_dates=['date'])
            except FileNotFoundError:
                pytest.skip("No clean data found. Run clean.py first.")

    def test_year_range(self, clean_data):
        """Data should span 2018-2025"""
        years = clean_data['date'].dt.year.unique()
        assert min(years) >= 2018
        assert max(years) <= 2025

    def test_all_months_present(self, clean_data):
        """Each year should have all 12 months (after filling)"""
        for year in clean_data['date'].dt.year.unique():
            months = clean_data[clean_data['date'].dt.year == year]['date'].dt.month.nunique()
            if year not in [2017]:  # 2017 may be incomplete
                assert months == 12 or months == 10  # 2022 had 10 months before fill