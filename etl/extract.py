"""
extract.py
-----------
Purpose: Pull the "Monthly tourist arrivals" table out of an SLTDA
Monthly Tourist Arrivals Report PDF and save it as a raw CSV.

This script does ONLY extraction. No cleaning of values, no calculations.
That happens later in transform.py. Keeping jobs separate makes each script
easier to test and debug on its own.

Usage:
    python etl/extract.py <input_pdf> <output_csv>
    
Examples:
    python etl/extract.py data/raw/sltda_dec2022.pdf data/raw/dec2022_raw.csv
    python etl/extract.py data/raw/sltda_dec2023.pdf data/raw/dec2023_raw.csv
    python etl/extract.py data/raw/sltda_dec2025.pdf data/raw/dec2025_raw.csv
"""

import sys
import re
import pdfplumber
import pandas as pd

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']


def find_table_page(pdf):
    """
    Find the page containing the monthly arrivals table.
    Searches for Table 1 or pages with month names and numbers.
    """
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        
        # Check for Table 1
        if 'Table 1' in text and ('Monthly tourist arrivals' in text or 'Monthly Tourist Arrivals' in text):
            print(f"Found 'Table 1' on page {i + 1}")
            return i
        
        # Check if page has month names and arrival numbers
        month_count = sum(1 for month in MONTHS if month in text)
        if month_count >= 3 and re.search(r'[\d,]+', text):
            print(f"Found arrivals data on page {i + 1} (found {month_count} months)")
            return i
    
    raise ValueError("Could not find the monthly arrivals table in this PDF.")


def clean_number(raw_value):
    """
    Clean numbers with spaces, commas, and special characters.
    Handles: '208, 253' -> 208253, '102,545' -> 102545
    """
    if not raw_value or raw_value.strip() == '':
        return None
    
    value = re.sub(r'\s+', '', str(raw_value))
    value = value.replace(',', '')
    value = value.replace('%', '')
    value = value.replace('(', '').replace(')', '')
    value = value.replace('*', '')
    value = value.strip()
    
    if value == '' or value == '-' or value == '--' or value == 'nan':
        return None
    
    try:
        return int(float(value))
    except ValueError:
        return None


def extract_table_from_text(text):
    """
    Extract the monthly arrivals table from the page text.
    Works for all years (2022, 2023, 2025).
    """
    lines = text.split('\n')
    
    # Find where the table starts
    start_idx = None
    
    # Look for "Table 1" or "Monthly tourist arrivals"
    for i, line in enumerate(lines):
        if 'Table 1' in line and ('Monthly tourist arrivals' in line or 'Monthly Tourist Arrivals' in line):
            start_idx = i + 1
            break
    
    # If not found, look for month names at the start of lines
    if start_idx is None:
        for i, line in enumerate(lines):
            line = line.strip()
            for month in MONTHS:
                if line.startswith(month) or (line.startswith(month[:4]) and len(line) > 4):
                    start_idx = i
                    break
            if start_idx is not None:
                break
    
    # If still not found, look for any line with month names
    if start_idx is None:
        for i, line in enumerate(lines):
            line = line.strip()
            for month in MONTHS:
                if month in line and (re.search(r'\d', line) or line.startswith(month)):
                    start_idx = i
                    break
            if start_idx is not None:
                break
    
    if start_idx is None:
        return None
    
    # Extract table rows
    table_rows = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # Check if this is a table row
        is_table_row = False
        
        # Check if it starts with a month
        for month in MONTHS:
            if line.startswith(month) or (line.startswith(month[:4]) and len(line) > 4):
                is_table_row = True
                break
        
        # Or has numbers and contains a month
        if not is_table_row:
            for month in MONTHS:
                if month in line and re.search(r'[\d,]+', line):
                    is_table_row = True
                    break
        
        # Or is TOTAL row
        if line.startswith('TOTAL'):
            is_table_row = True
        
        if is_table_row:
            table_rows.append(line)
        elif table_rows and line and not re.search(r'[\d,]', line) and len(line) > 15:
            # Stop if we hit a long text line without numbers
            break
    
    if not table_rows:
        return None
    
    # Parse the rows
    data_rows = []
    for line in table_rows:
        # Try splitting by 2+ spaces first
        parts = re.split(r'\s{2,}', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        # If that didn't work, split by single spaces
        if len(parts) < 3:
            parts = line.split()
            parts = [p.strip() for p in parts if p.strip()]
        
        if not parts:
            continue
        
        # Check if this is a month row
        month = None
        for m in MONTHS:
            if parts[0] == m or parts[0].startswith(m[:4]) or m in parts[0]:
                month = m
                break
        
        if month:
            data_rows.append([month] + parts[1:])
        elif parts[0] == 'TOTAL':
            data_rows.append(parts)
    
    return data_rows


def extract_arrivals_table(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page_index = find_table_page(pdf)
        page = pdf.pages[page_index]
        
        # Extract text from the page
        text = page.extract_text() or ''
        
        print(f"\n--- Page {page_index + 1} text preview (first 600 chars) ---")
        print(text[:600])
        print("--- End of preview ---\n")
        
        # Try to extract table from text
        data_rows = extract_table_from_text(text)
        
        if data_rows is None or not data_rows:
            raise ValueError("Could not extract table from page text")
        
        print(f"Found {len(data_rows)} data rows")
        
        # Extract years from the text
        years = re.findall(r'20\d{2}', text)
        years = list(dict.fromkeys(years))
        
        if len(years) >= 2:
            # Sort years ascending and take first two
            years = sorted(years, key=lambda x: int(x))
            year1 = int(years[0])
            year2 = int(years[1])
            print(f"Using years from text: {year1} and {year2}")
        else:
            # Fallback: use the filename
            filename = pdf_path.split('/')[-1] if '/' in pdf_path else pdf_path.split('\\')[-1]
            year_match = re.search(r'20\d{2}', filename)
            if year_match:
                main_year = int(year_match.group())
                year1 = main_year - 1
                year2 = main_year
                print(f"Using years from filename: {year1} and {year2}")
            else:
                # Last resort: use 2022 and 2023
                print("Warning: Could not determine years, using 2022 and 2023")
                year1, year2 = 2022, 2023
        
        # Parse data rows
        records = []
        for row in data_rows:
            if not row:
                continue
            
            month = row[0].strip()
            
            # Stop at TOTAL
            if month == 'TOTAL':
                break
            
            # Check if it's a valid month
            if month not in MONTHS:
                # Try to find month name in the string
                found = False
                for m in MONTHS:
                    if m in month or month in m or month.startswith(m[:4]):
                        month = m
                        found = True
                        break
                if not found:
                    # Try first part if it's a word
                    first_word = month.split()[0] if ' ' in month else month
                    for m in MONTHS:
                        if first_word.startswith(m[:4]) or m.startswith(first_word[:4]):
                            month = m
                            found = True
                            break
                if not found:
                    print(f"Skipping unknown row: {row}")
                    continue
            
            # Extract numeric values
            numeric_values = []
            for cell in row[1:]:
                clean_val = clean_number(cell)
                if clean_val is not None:
                    numeric_values.append(clean_val)
            
            # We expect at least 2 values (arrivals for each year)
            if len(numeric_values) >= 2:
                records.append({'month': month, 'year': year1, 'arrivals': numeric_values[0]})
                records.append({'month': month, 'year': year2, 'arrivals': numeric_values[1]})
            elif len(numeric_values) == 1:
                records.append({'month': month, 'year': year1, 'arrivals': numeric_values[0]})
                print(f"Warning: Only 1 value found for {month}, using only year {year1}")
        
        if not records:
            raise ValueError("No valid data rows found")
        
        return pd.DataFrame(records)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python etl/extract.py <input_pdf> <output_csv>")
        print("\nExamples:")
        print("  python etl/extract.py data/raw/sltda_dec2022.pdf data/raw/dec2022_raw.csv")
        print("  python etl/extract.py data/raw/sltda_dec2023.pdf data/raw/dec2023_raw.csv")
        print("  python etl/extract.py data/raw/sltda_dec2025.pdf data/raw/dec2025_raw.csv")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_csv = sys.argv[2]

    try:
        df = extract_arrivals_table(input_pdf)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Extracted {len(df)} rows -> {output_csv}")
        print("\nSample data (first 6 rows):")
        print(df.head(6))
        print("\nData summary:")
        print(df.groupby('year')['arrivals'].agg(['count', 'min', 'max', 'sum']))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)