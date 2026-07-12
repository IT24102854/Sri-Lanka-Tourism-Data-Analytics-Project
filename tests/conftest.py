"""
tests/conftest.py
-----------------
Pytest configuration and fixtures.
"""

import sys
import os
import pytest
import unittest.mock as mock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock heavy dependencies for testing
import sys
sys.modules['pdfplumber'] = mock.Mock()
sys.modules['prophet'] = mock.Mock()
sys.modules['sklearn'] = mock.Mock()

@pytest.fixture
def sample_data():
    """Sample data for testing"""
    import pandas as pd
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
        'arrivals': [100000 + i * 5000 for i in range(12)]
    })