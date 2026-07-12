# 🇱🇰 Sri Lanka Tourism Analytics

> **End-to-end data pipeline analyzing 8 years of tourism data (2018-2025)**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](http://localhost:8501)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://postgresql.org)

---

## 📊 **Project Overview**

This project analyzes Sri Lanka's tourism recovery journey from the **2019 Easter attacks** through **COVID-19** to the **record-breaking 2025 season**. It includes a complete data pipeline from PDF extraction to interactive dashboard.

### 🔥 **Key Results**

| Metric | Result |
|--------|--------|
| **2026 Forecast** | 2.59M tourists (+9.6% growth) |
| **Model Accuracy** | 11.0% MAPE (good for tourism) |
| **Best Year** | 2025 (2.36M arrivals - NEW RECORD!) |
| **Worst Year** | 2021 (0.19M arrivals - COVID) |
| **Peak Month** | December (308K avg) |
| **Data Points** | 96 months (8 years) |

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PDF Reports  →  Extract  →  Clean  →  PostgreSQL  →  Dashboard│
│      ↓             ↓          ↓           ↓             ↓      │
│   SLTDA       pdfplumber   Pandas    psycopg2     Streamlit    │
│   2018-2025     Python      Python      SQL       Plotly       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Tech Stack**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Extraction** | Python, pdfplumber | Extract tables from PDFs |
| **Data Processing** | Pandas, NumPy | Clean & transform data |
| **Forecasting** | Prophet (Meta) | Time series forecasting |
| **Database** | PostgreSQL | Data storage |
| **Dashboard** | Streamlit, Plotly | Interactive visualization |
| **Testing** | Pytest | Unit testing |

---

## 📁 **Project Structure**

```
sl-tourism-project/
├── dashboard/                    # Streamlit dashboard
│   └── app.py                   # Main dashboard application
├── data/
│   ├── raw/                     # Original PDFs & extracted CSVs
│   │   ├── sltda_dec*.pdf      # Source PDF reports
│   │   └── dec*_raw.csv        # Extracted raw data
│   └── processed/               # Cleaned data & forecasts
│       ├── arrivals_clean.csv   # Clean time series (2018-2025)
│       ├── forecast_future.csv  # 2026 forecast
│       └── *.png                # Generated visualizations
├── etl/                         # ETL pipeline
│   ├── extract.py              # PDF → CSV extraction
│   ├── clean.py                # Combine & deduplicate
│   └── load.py                 # PostgreSQL loader
├── notebooks/                   # Jupyter notebooks (6 total)
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_data_cleaning.ipynb  # Data cleaning process
│   ├── 03_feature_engineering.ipynb  # Feature creation
│   ├── 04_modeling_forecast.ipynb    # Prophet model + viz
│   ├── 05_model_comparison.ipynb     # Model comparison
│   └── 06_results_dashboard.ipynb    # Final results
├── src/                         # Source code
│   ├── model.py                # Prophet forecasting
│   └── compare_models.py       # Multi-model comparison
├── sql/                         # Database schema
│   └── schema.sql              # PostgreSQL table definitions
├── tests/                       # Unit tests (19 passing)
│   ├── test_etl.py             # ETL tests
│   └── conftest.py             # Test configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.10+
- PostgreSQL 16+

### **Installation**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sl-tourism-project.git
cd sl-tourism-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# 5. Set up PostgreSQL
psql -U postgres -c "CREATE DATABASE tourism_db;"
psql -U postgres -c "CREATE USER tourism_user WITH PASSWORD 'tourism_pass123';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE tourism_db TO tourism_user;"
psql -U tourism_user -d tourism_db -f sql/schema.sql

# 6. Load data
python etl/load.py

# 7. Run dashboard
streamlit run dashboard/app.py

# 8. Open browser
# Go to: http://localhost:8501
```

---

## 📊 **Data Pipeline Details**

### **1. Extraction (`etl/extract.py`)**

```bash
# Extract data from PDF reports
python etl/extract.py data/raw/sltda_dec2025.pdf data/raw/dec2025_raw.csv
```

**What it does:**
- Reads SLTDA PDF reports (2018-2025)
- Extracts monthly tourist arrival tables
- Handles different PDF formats
- Saves as CSV files

### **2. Cleaning (`etl/clean.py`)**

```bash
# Combine and clean all raw files
python etl/clean.py data/raw/dec*.csv data/processed/arrivals_clean.csv
```

**What it does:**
- Combines all CSV files
- Deduplicates overlapping data
- Handles missing months
- Saves clean dataset

### **3. Forecasting (`src/model.py`)**

```bash
# Train Prophet model and generate forecast
python src/model.py
```

**What it does:**
- Prophet model with tuned parameters
- 12-month holdout evaluation
- 80% confidence intervals
- 2026 forecast generation

### **4. Dashboard (`dashboard/app.py`)**

```bash
# Run interactive dashboard
streamlit run dashboard/app.py
```

**What it does:**
- Real-time PostgreSQL queries
- Interactive visualizations
- YoY heatmap
- Confidence intervals
- Data export

---

## 📈 **Results**

### **Recovery Story**

```
2018: 2.33M 🏆 Pre-COVID Peak
2019: 1.91M 💔 Easter Attacks (-18%)
2020: 0.51M 🦠 COVID Collapse (-73.5%)
2021: 0.19M ⚠️ Border Closures (-61.7%)
2022: 0.57M 📈 Recovery Begins (+192%)
2023: 1.49M 🚀 Strong Recovery (+162%)
2024: 2.05M ✅ Near Normal (+38%)
2025: 2.36M 🏆 NEW RECORD! (+15%)
2026: 2.59M 🔮 Forecast (+9.6%)
```

### **2026 Forecast**

| Month | Predicted | 80% CI |
|-------|-----------|--------|
| January | 241,930 | 215,516 - 269,745 |
| February | 264,012 | 233,340 - 293,049 |
| March | 226,772 | 199,950 - 254,492 |
| April | 180,535 | 152,766 - 209,398 |
| May | 134,793 | 107,015 - 164,972 |
| June | 176,581 | 147,505 - 207,253 |
| July | 214,782 | 185,264 - 246,067 |
| August | 218,459 | 187,302 - 249,668 |
| September | 186,374 | 152,722 - 221,261 |
| October | 199,010 | 164,236 - 233,997 |
| November | 237,910 | 202,314 - 274,249 |
| December | 308,571 | 271,288 - 345,190 |

### **Model Performance**

| Model | MAPE | RMSE | MAE |
|-------|------|------|-----|
| **Prophet** | **11.0%** | **22,796** | **19,495** |
| SARIMA | 40.1% | 74,445 | 70,740 |
| Linear Regression | 53.0% | 107,166 | 103,303 |

**Prophet outperformed SARIMA by 3.4x in MAPE**, confirming its suitability for this dataset.

---

## 🧪 **Testing**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=etl --cov=src --cov-report=html

# Run specific test
pytest tests/test_etl.py::TestCleanNumber -v
```

**Results:** 19 tests passing ✅

---

## 📊 **Dashboard Features**

### **1. Key Metrics Dashboard**
- Total arrivals, average monthly, peak month
- YoY growth indicators
- Interactive year selector

### **2. 📈 Forecast with Confidence Intervals**
- Historical data (2018-2025) + 2026 forecast
- 80% confidence interval visualization
- Error bars showing prediction uncertainty

### **3. 🔥 Year-over-Year Heatmap**
- Monthly arrivals across all years
- Growth rate heatmap
- Color-coded performance visualization

### **4. 🎯 What-if Scenario Simulator**
- Adjust growth rates (-10% to +30%)
- Economic scenarios (Pessimistic/Baseline/Optimistic)
- Holiday impact scenarios
- Shock scenarios (minor/major disruptions)
- Real-time scenario comparison

### **5. 🏆 Benchmarking**
- Compare Sri Lanka vs 7 other destinations
- Recovery rates vs pre-COVID
- Global ranking metrics
- Key observations

### **6. 📋 Forecast Details**
- Monthly breakdown with confidence bounds
- Peak and low month identification
- Data export (CSV download)
---

## 📋 **Notebooks**

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_data_cleaning.ipynb` | Data cleaning process |
| `03_feature_engineering.ipynb` | Feature creation |
| `04_modeling_forecast.ipynb` | Prophet model + visualization |
| `05_model_comparison.ipynb` | Model comparison |
| `06_results_dashboard.ipynb` | Final results |

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📝 **License**

MIT License - see [LICENSE](LICENSE) for details

---

## 📧 **Contact**

- **Name:** M.R.C.D.Bandara
- **Email:**chithminibandara@gmail.com
- **LinkedIn:** [your-linkedin](https://linkedin.com/in/your-profile)
- **GitHub:** [your-username](https://github.com/IT24102854)

---

## ⭐ **Acknowledgments**

- SLTDA for providing the data
- Meta for the Prophet library
- Open-source community for all the tools

---

## 📚 **References**

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## 🏆 **Portfolio Impact**

This project demonstrates:

| Skill | Evidence |
|-------|----------|
| **Data Engineering** | Complete ETL pipeline from PDF to database |
| **Data Science** | Time series forecasting with Prophet |
| **Data Visualization** | Interactive Streamlit dashboard |
| **Database Design** | PostgreSQL schema with proper constraints |
| **Testing** | 19 passing unit tests |
| **Communication** | Professional documentation |

---

**⭐ Star this repo if you find it useful!**

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

---

## ✅ **README Checklist**

| Section | Status |
|---------|--------|
| Project Title & Badges | ✅ |
| Project Overview | ✅ |
| Key Results | ✅ |
| Architecture Diagram | ✅ |
| Tech Stack | ✅ |
| Project Structure | ✅ |
| Quick Start Guide | ✅ |
| Pipeline Details | ✅ |
| Results & Forecast | ✅ |
| Model Performance | ✅ |
| Testing | ✅ |
| Dashboard Features | ✅ |
| Notebooks | ✅ |
| Contributing | ✅ |
| License | ✅ |
| Contact | ✅ |
| Portfolio Impact | ✅ |

---

## 🚀 **How to Add This README**

```bash
# 1. Replace your README.md
# Copy the entire content above into README.md

# 2. Update the placeholders
# Replace "yourusername" with your GitHub username
# Replace "your.email@example.com" with your email

# 3. Commit and push
git add README.md
git commit -m "📝 Update README with complete project documentation"
git push origin main
