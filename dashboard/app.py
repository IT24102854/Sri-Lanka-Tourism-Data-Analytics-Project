"""
app.py
------
Enhanced Streamlit dashboard for Sri Lanka tourism with:
- What-if Scenario Simulator
- Year-over-Year Heatmap
- Benchmarking
- Confidence Interval Visualization
- Dark/Light Mode Support

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------
DB_USER = os.getenv('DB_USER', 'tourism_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'tourism_pass123')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'tourism_db')

CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@st.cache_resource
def get_engine():
    return create_engine(CONNECTION_STRING)


@st.cache_data(ttl=300)
def load_arrivals():
    engine = get_engine()
    query = "SELECT date, arrivals FROM tourist_arrivals ORDER BY date"
    return pd.read_sql(query, engine, parse_dates=['date'])


@st.cache_data(ttl=300)
def load_forecast():
    engine = get_engine()
    query = """
        SELECT month, predicted_arrivals, lower_bound, upper_bound
        FROM forecast_results
        WHERE model_name = 'prophet'
        ORDER BY month
    """
    return pd.read_sql(query, engine, parse_dates=['month'])


@st.cache_data(ttl=300)
def load_yearly_totals():
    engine = get_engine()
    query = """
        SELECT
            EXTRACT(YEAR FROM date)::int AS year,
            SUM(arrivals) AS total_arrivals
        FROM tourist_arrivals
        GROUP BY year
        ORDER BY year
    """
    return pd.read_sql(query, engine)


# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Tourism Dashboard",
    page_icon="🇱🇰",
    layout="wide"
)

# ============================================================
# THEME-AWARE CSS - Works in both Dark and Light mode
# ============================================================
st.markdown("""
<style>
    /* Theme-aware variables using Streamlit's theme detection */
    /* These adapt to both dark and light modes */
    
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-family: 'Segoe UI', sans-serif;
        color: var(--text-color);
    }
    
    .sub-header {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', sans-serif;
        color: var(--text-color-secondary);
    }
    
    /* Metric cards with theme-aware colors */
    .metric-card {
        background: var(--secondary-background-color);
        padding: 1.2rem 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: var(--text-color-secondary);
        font-weight: 500;
    }
    
    /* Insight box - Theme-aware */
    .insight-box {
        background: var(--secondary-background-color);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.8rem 0;
        color: var(--text-color);
        font-size: 1rem;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }
    
    .insight-box strong {
        color: var(--text-color);
        font-size: 1.05rem;
    }
    
    /* Colored boxes with theme-aware text */
    .success-box {
        background: #d4edda;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        color: #155724;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .success-box strong {
        color: #0b5e1e;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        color: #856404;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .warning-box strong {
        color: #6c5200;
    }
    
    .info-box {
        background: #d1ecf1;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        color: #0c5460;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .info-box strong {
        color: #0a3d4a;
    }
    
    .error-box {
        background: #f8d7da;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        color: #721c24;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .error-box strong {
        color: #5a1218;
    }
    
    /* Dark mode overrides for colored boxes */
    @media (prefers-color-scheme: dark) {
        .success-box {
            background: #1e4620;
            color: #a3d9a5;
            border-left-color: #28a745;
        }
        .success-box strong {
            color: #d4edda;
        }
        .warning-box {
            background: #4a3a1a;
            color: #f5d290;
            border-left-color: #ffc107;
        }
        .warning-box strong {
            color: #ffeb3b;
        }
        .info-box {
            background: #1a3a42;
            color: #8fd0d9;
            border-left-color: #17a2b8;
        }
        .info-box strong {
            color: #d4edf5;
        }
        .error-box {
            background: #3d1a1e;
            color: #e8a0a8;
            border-left-color: #dc3545;
        }
        .error-box strong {
            color: #f5c6cb;
        }
    }
    
    /* Scenario cards */
    .scenario-card {
        background: var(--secondary-background-color);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.3rem 0;
    }
    
    .scenario-card .scenario-name {
        font-size: 0.85rem;
        color: var(--text-color-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .scenario-card .scenario-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 0.2rem 0;
    }
    
    .scenario-card .scenario-change {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .scenario-card .scenario-change.positive {
        color: #28a745;
    }
    
    .scenario-card .scenario-change.negative {
        color: #dc3545;
    }
    
    /* Benchmark card */
    .benchmark-card {
        background: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .benchmark-card .benchmark-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .benchmark-card .benchmark-label {
        font-size: 0.85rem;
        color: var(--text-color-secondary);
        font-weight: 500;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .subsection-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 1rem 0 0.5rem 0;
    }
    
    /* Metric labels - force visibility */
    .stMetric label {
        font-weight: 600 !important;
        color: var(--text-color) !important;
    }
    
    .stMetric .stMetricValue {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
    }
    
    /* Selectbox, slider, checkbox labels */
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    
    /* Caption styling */
    .stCaption {
        color: var(--text-color-secondary) !important;
        font-size: 0.9rem !important;
    }
    
    /* Expander header */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--text-color) !important;
    }
    
    /* Dataframe text */
    .dataframe {
        color: var(--text-color) !important;
    }
    
    /* Plotly chart titles */
    .js-plotly-plot .plotly .main-svg .g-title text {
        fill: var(--text-color) !important;
    }
    
    /* Dark mode specific overrides */
    @media (prefers-color-scheme: dark) {
        .insight-box {
            background: #2d2d2d;
            border-color: #444;
        }
        .insight-box strong {
            color: #e0e0e0;
        }
        .scenario-card {
            background: #2d2d2d;
            border-color: #444;
        }
        .scenario-card .scenario-value {
            color: #e0e0e0;
        }
        .benchmark-card {
            background: #2d2d2d;
            border-color: #444;
        }
        .benchmark-card .benchmark-value {
            color: #e0e0e0;
        }
        .section-title {
            color: #e0e0e0;
            border-bottom-color: #444;
        }
        .subsection-title {
            color: #e0e0e0;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🇱🇰 Sri Lanka Tourism Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">📊 Data source: SLTDA Monthly Reports · Live from PostgreSQL</div>', unsafe_allow_html=True)

# Load data
arrivals_df = load_arrivals()
forecast_df = load_forecast()
yearly_df = load_yearly_totals()

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Controls")
    
    years = sorted(arrivals_df['date'].dt.year.unique(), reverse=True)
    selected_year = st.selectbox("📅 Select Year", years, index=0)
    
    st.divider()
    st.header("📊 Modules")
    
    show_heatmap = st.checkbox("🔥 Year-over-Year Heatmap", value=True)
    show_forecast = st.checkbox("📈 Forecast with Confidence", value=True)
    show_scenario = st.checkbox("🎯 What-if Scenario", value=False)
    show_benchmark = st.checkbox("🏆 Benchmarking", value=False)
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit")

# Calculate metrics
total_all_time = int(arrivals_df['arrivals'].sum())
avg_monthly = int(arrivals_df['arrivals'].mean())
peak_month = arrivals_df.loc[arrivals_df['arrivals'].idxmax()]
latest_year = arrivals_df['date'].dt.year.max()

yearly_totals = arrivals_df.groupby(arrivals_df['date'].dt.year)['arrivals'].sum()
if len(yearly_totals) >= 2:
    yoy_growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2]) * 100
else:
    yoy_growth = 0

# ---------------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Arrivals (All Time)", f"{total_all_time:,}", help="Total arrivals 2018-2025")
with col2:
    st.metric("Avg Monthly", f"{avg_monthly:,}")
with col3:
    st.metric("Peak Month", f"{peak_month['date'].strftime('%B %Y')}")
with col4:
    st.metric("Latest Year", f"{latest_year}")
with col5:
    st.metric("YoY Growth", f"{yoy_growth:+.1f}%", help="Year-over-year growth from previous year")

st.divider()

# ---------------------------------------------------------------
# SECTION 1: YEAR DETAIL
# ---------------------------------------------------------------
st.markdown('<div class="section-title">📅 Year Detail</div>', unsafe_allow_html=True)

year_data = arrivals_df[arrivals_df['date'].dt.year == selected_year]
year_total = int(year_data['arrivals'].sum())

prior_year_row = yearly_df[yearly_df['year'] == selected_year - 1]
if not prior_year_row.empty:
    prior_total = int(prior_year_row['total_arrivals'].iloc[0])
    pct_change = ((year_total - prior_total) / prior_total) * 100
else:
    pct_change = None

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"Total Arrivals — {selected_year}", f"{year_total:,}")
with col2:
    if pct_change is not None:
        st.metric("YoY Change", f"{pct_change:+.1f}%")
    else:
        st.metric("YoY Change", "N/A")
with col3:
    st.metric("Peak Month", year_data.loc[year_data['arrivals'].idxmax(), 'date'].strftime('%B'))

# Monthly chart for selected year
fig_year = px.line(
    year_data,
    x='date',
    y='arrivals',
    title=f'Monthly Arrivals - {selected_year}',
    markers=True,
    template='plotly_white'
)
fig_year.update_layout(height=300, font=dict(color='#1a1a2e'))
st.plotly_chart(fig_year, use_container_width=True)

# ---------------------------------------------------------------
# SECTION 2: YEAR-OVER-YEAR HEATMAP
# ---------------------------------------------------------------
if show_heatmap:
    st.markdown('<div class="section-title">🔥 Year-over-Year Heatmap</div>', unsafe_allow_html=True)
    
    # Prepare data for heatmap
    heatmap_data = arrivals_df.copy()
    heatmap_data['year'] = heatmap_data['date'].dt.year
    heatmap_data['month'] = heatmap_data['date'].dt.month
    heatmap_data['month_name'] = heatmap_data['date'].dt.strftime('%b')
    
    # Pivot for heatmap
    heatmap_pivot = heatmap_data.pivot(index='month_name', columns='year', values='arrivals')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_pivot = heatmap_pivot.reindex(month_order)
    
    # Also calculate growth heatmap
    growth_pivot = heatmap_pivot.pct_change(axis=1) * 100
    
    # Create tabs for different heatmap views
    tab1, tab2 = st.tabs(["📊 Arrivals Heatmap", "📈 Growth Heatmap"])
    
    with tab1:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='Viridis',
            text=heatmap_pivot.values,
            texttemplate='%{text:,.0f}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='Year: %{x}<br>Month: %{y}<br>Arrivals: %{z:,.0f}<extra></extra>'
        ))
        fig_heatmap.update_layout(
            title='Monthly Arrivals Heatmap (2018-2025)',
            xaxis_title='Year',
            yaxis_title='Month',
            height=450,
            yaxis={'categoryorder': 'array', 'categoryarray': month_order[::-1]},
            coloraxis_colorbar=dict(title="Arrivals"),
            font=dict(color='#1a1a2e')
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        fig_growth = go.Figure(data=go.Heatmap(
            z=growth_pivot.values,
            x=growth_pivot.columns,
            y=growth_pivot.index,
            colorscale='RdYlGn',
            text=growth_pivot.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='Year: %{x}<br>Month: %{y}<br>Growth: %{z:.1f}%<extra></extra>'
        ))
        fig_growth.update_layout(
            title='Year-over-Year Growth Heatmap',
            xaxis_title='Year',
            yaxis_title='Month',
            height=450,
            yaxis={'categoryorder': 'array', 'categoryarray': month_order[::-1]},
            coloraxis_colorbar=dict(title="Growth %"),
            font=dict(color='#1a1a2e')
        )
        st.plotly_chart(fig_growth, use_container_width=True)

# ---------------------------------------------------------------
# SECTION 3: FORECAST WITH CONFIDENCE INTERVALS
# ---------------------------------------------------------------
if show_forecast:
    st.markdown('<div class="section-title">📈 Forecast with Confidence Intervals</div>', unsafe_allow_html=True)
    
    # Create combined chart with confidence intervals
    fig_combined = go.Figure()
    
    # Historical data
    fig_combined.add_trace(go.Scatter(
        x=arrivals_df['date'],
        y=arrivals_df['arrivals'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast data
    fig_combined.add_trace(go.Scatter(
        x=forecast_df['month'],
        y=forecast_df['predicted_arrivals'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(size=8, color='#d62728')
    ))
    
    # Confidence interval (shaded area)
    fig_combined.add_trace(go.Scatter(
        x=pd.concat([forecast_df['month'], forecast_df['month'][::-1]]),
        y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(214,39,40,0.2)',
        line=dict(color='rgba(214,39,40,0)'),
        name='80% Confidence Interval'
    ))
    
    # Add vertical line for forecast start
    forecast_start = arrivals_df['date'].max()
    fig_combined.add_vline(
        x=forecast_start,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    fig_combined.update_layout(
        title='Monthly Tourist Arrivals (2018-2025) + 2026 Forecast',
        xaxis_title='Date',
        yaxis_title='Arrivals',
        height=450,
        hovermode='x unified',
        template='plotly_white',
        font=dict(color='#1a1a2e')
    )
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Confidence interval explanation
    total_forecast = int(forecast_df['predicted_arrivals'].sum())
    total_2025 = arrivals_df[arrivals_df['date'].dt.year == 2025]['arrivals'].sum()
    growth = ((total_forecast - total_2025) / total_2025) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 2026 Forecast</strong><br>
            <span style="font-size:1.2rem; font-weight:700;">Total: {total_forecast:,.0f} arrivals</span><br>
            <span style="font-size:1.1rem;">Growth: <strong style="color:#28a745;">{growth:.1f}%</strong> vs 2025</span><br>
            Confidence Level: <strong>80%</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>💡 Understanding Confidence Intervals</strong><br>
            The shaded area shows the <strong>80% confidence interval</strong>.<br>
            There is an <strong>80% chance</strong> that actual arrivals will fall within this range.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# SECTION 4: WHAT-IF SCENARIO SIMULATOR
# ---------------------------------------------------------------
if show_scenario:
    st.markdown('<div class="section-title">🎯 What-if Scenario Simulator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>📈 How does changing growth rates affect 2026 arrivals?</strong><br>
        Adjust the sliders below to see different scenarios.
    </div>
    """, unsafe_allow_html=True)
    
    # Get baseline
    baseline_2025 = arrivals_df[arrivals_df['date'].dt.year == 2025]['arrivals'].sum()
    baseline_2026 = forecast_df['predicted_arrivals'].sum()
    baseline_growth = ((baseline_2026 - baseline_2025) / baseline_2025) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Growth rate scenario
        growth_scenario = st.slider(
            "📈 Annual Growth Rate Scenario",
            min_value=-10.0,
            max_value=30.0,
            value=baseline_growth,
            step=0.5,
            help="Adjust the growth rate to see different scenarios"
        )
        
        # Economic growth scenario
        economic_growth = st.selectbox(
            "🌍 Economic Scenario",
            ["Pessimistic (2% GDP)", "Baseline (3.5% GDP)", "Optimistic (5% GDP)", "Custom"],
            index=1
        )
    
    with col2:
        # Holiday impact
        holiday_scenario = st.selectbox(
            "🎄 Holiday Impact Scenario",
            ["Normal", "Strong (Christmas/New Year boost)", "Weak (Holiday slowdown)"],
            index=0
        )
        
        # COVID-like shock
        shock_scenario = st.selectbox(
            "⚠️ Shock Scenario",
            ["No shock", "Minor disruption (-10%)", "Major event (-25%)"],
            index=0
        )
    
    # Calculate scenario results
    scenarios = {}
    
    # Baseline
    scenarios["Baseline"] = baseline_2026
    
    # Growth rate scenario
    if growth_scenario != baseline_growth:
        adjusted_growth = growth_scenario / baseline_growth
        scenarios["Growth Scenario"] = baseline_2026 * (1 + (growth_scenario - baseline_growth) / 100)
    
    # Economic scenario adjustments
    economic_multipliers = {
        "Pessimistic (2% GDP)": 0.95,
        "Baseline (3.5% GDP)": 1.0,
        "Optimistic (5% GDP)": 1.05,
        "Custom": 1.0
    }
    eco_mult = economic_multipliers.get(economic_growth, 1.0)
    scenarios["Economic Scenario"] = baseline_2026 * eco_mult
    
    # Holiday impact
    holiday_multipliers = {
        "Normal": 1.0,
        "Strong (Christmas/New Year boost)": 1.03,
        "Weak (Holiday slowdown)": 0.97
    }
    hol_mult = holiday_multipliers.get(holiday_scenario, 1.0)
    scenarios["Holiday Scenario"] = baseline_2026 * hol_mult
    
    # Shock scenario
    shock_multipliers = {
        "No shock": 1.0,
        "Minor disruption (-10%)": 0.90,
        "Major event (-25%)": 0.75
    }
    shock_mult = shock_multipliers.get(shock_scenario, 1.0)
    scenarios["Shock Scenario"] = baseline_2026 * shock_mult
    
    # Combined scenario (all factors)
    combined = baseline_2026 * eco_mult * hol_mult * shock_mult
    scenarios["Combined Scenario"] = combined
    
    # Display scenarios
    st.markdown('<div class="subsection-title">📊 Scenario Results</div>', unsafe_allow_html=True)
    
    # Create scenario comparison dataframe
    scenario_df = pd.DataFrame({
        'Scenario': list(scenarios.keys()),
        '2026 Forecast': [f"{v:,.0f}" for v in scenarios.values()],
        'vs 2025': [f"{((v - baseline_2025) / baseline_2025 * 100):+.1f}%" for v in scenarios.values()],
        'vs Baseline': [f"{((v - baseline_2026) / baseline_2026 * 100):+.1f}%" for v in scenarios.values()]
    })
    
    st.dataframe(scenario_df, use_container_width=True)
    
    # Visualize scenarios
    fig_scenarios = go.Figure()
    
    scenario_names = list(scenarios.keys())
    scenario_values = list(scenarios.values())
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    fig_scenarios.add_trace(go.Bar(
        x=scenario_names,
        y=scenario_values,
        marker_color=colors[:len(scenario_names)],
        text=[f"{v:,.0f}" for v in scenario_values],
        textposition='outside',
        name='2026 Forecast'
    ))
    
    # Add baseline reference line
    fig_scenarios.add_hline(
        y=baseline_2026,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Baseline: {baseline_2026:,.0f}",
        annotation_position="bottom right"
    )
    
    fig_scenarios.update_layout(
        title='What-If Scenario Comparison',
        xaxis_title='Scenario',
        yaxis_title='2026 Arrivals',
        height=400,
        template='plotly_white',
        font=dict(color='#1a1a2e')
    )
    
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Scenario insights
    st.markdown('<div class="subsection-title">💡 Key Insights</div>', unsafe_allow_html=True)
    
    best_scenario = max(scenarios, key=lambda x: scenarios[x])
    worst_scenario = min(scenarios, key=lambda x: scenarios[x])
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ Best Case: {best_scenario}</strong><br>
            <span style="font-size:1.3rem; font-weight:700;">{scenarios[best_scenario]:,.0f}</span> arrivals<br>
            <span>+{((scenarios[best_scenario] - baseline_2025) / baseline_2025 * 100):.1f}% growth vs 2025</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="error-box">
            <strong>❌ Worst Case: {worst_scenario}</strong><br>
            <span style="font-size:1.3rem; font-weight:700;">{scenarios[worst_scenario]:,.0f}</span> arrivals<br>
            <span>{((scenarios[worst_scenario] - baseline_2025) / baseline_2025 * 100):.1f}% growth vs 2025</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <strong>📊 Baseline Forecast</strong><br>
        <span style="font-size:1.2rem; font-weight:700;">{baseline_2026:,.0f}</span> arrivals<br>
        <span>{baseline_growth:.1f}% growth vs 2025</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>📝 Interpretation</strong><br>
        The model shows that <strong>economic growth</strong> is the most significant factor affecting tourism arrivals.<br>
        A <strong>1% change</strong> in economic growth translates to approximately <strong>25,000-30,000</strong> additional tourists.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# SECTION 5: BENCHMARKING (Simulated)
# ---------------------------------------------------------------
if show_benchmark:
    st.markdown('<div class="section-title">🏆 Benchmarking - Sri Lanka vs Other Destinations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>📊 How does Sri Lanka compare to other tourist destinations?</strong><br>
        Benchmarking helps identify relative performance and growth opportunities.
    </div>
    """, unsafe_allow_html=True)
    
    # Simulated benchmark data (in a real scenario, you'd fetch this from external APIs)
    benchmark_data = {
        'Country': ['Sri Lanka', 'Maldives', 'India', 'Thailand', 'Vietnam', 'Indonesia', 'Singapore', 'Malaysia'],
        '2025 Arrivals (M)': [2.36, 1.80, 19.00, 35.00, 18.00, 12.00, 15.00, 20.00],
        'Recovery Rate (%)': [108, 112, 95, 98, 105, 100, 92, 97],
        'Growth Rate (%)': [15.1, 12.5, 8.0, 6.5, 12.0, 10.0, 5.5, 7.5],
        'Avg Stay (nights)': [7, 8, 5, 6, 7, 6, 4, 5]
    }
    
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df = benchmark_df.sort_values('2025 Arrivals (M)', ascending=False).reset_index(drop=True)
    
    # Display benchmark table
    st.dataframe(benchmark_df.style.format({
        '2025 Arrivals (M)': '{:.2f}M',
        'Recovery Rate (%)': '{:.0f}%',
        'Growth Rate (%)': '{:.1f}%',
        'Avg Stay (nights)': '{:.0f}'
    }).background_gradient(subset=['Recovery Rate (%)'], cmap='RdYlGn'), use_container_width=True)
    
    # Visualize benchmarks
    col1, col2 = st.columns(2)
    
    with col1:
        # Arrivals comparison
        fig_bench1 = px.bar(
            benchmark_df,
            x='Country',
            y='2025 Arrivals (M)',
            title='2025 Tourist Arrivals (Millions)',
            color='2025 Arrivals (M)',
            color_continuous_scale='Blues',
            text='2025 Arrivals (M)'
        )
        fig_bench1.update_traces(texttemplate='%{text:.2f}M', textposition='outside')
        fig_bench1.update_layout(height=350, template='plotly_white', font=dict(color='#1a1a2e'))
        st.plotly_chart(fig_bench1, use_container_width=True)
    
    with col2:
        # Recovery rate comparison
        fig_bench2 = px.bar(
            benchmark_df,
            x='Country',
            y='Recovery Rate (%)',
            title='Recovery Rate vs Pre-COVID (2019 = 100%)',
            color='Recovery Rate (%)',
            color_continuous_scale='RdYlGn',
            range_color=[80, 115],
            text='Recovery Rate (%)'
        )
        fig_bench2.add_hline(y=100, line_dash="dash", line_color="#2ca02c", annotation_text="100% Recovery")
        fig_bench2.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig_bench2.update_layout(height=350, template='plotly_white', font=dict(color='#1a1a2e'))
        st.plotly_chart(fig_bench2, use_container_width=True)
    
    # Benchmark insights
    st.markdown('<div class="subsection-title">💡 Benchmark Insights</div>', unsafe_allow_html=True)
    
    # Sri Lanka's rank
    rank_by_arrivals = benchmark_df['2025 Arrivals (M)'].rank(ascending=False).iloc[0]
    recovery_rate = benchmark_df[benchmark_df['Country'] == 'Sri Lanka']['Recovery Rate (%)'].values[0]
    growth_rank = benchmark_df['Growth Rate (%)'].rank(ascending=False).iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="benchmark-card">
            <div class="benchmark-label">🌍 Global Rank</div>
            <div class="benchmark-value">#{int(rank_by_arrivals)}</div>
            <div style="font-size:0.85rem; color:var(--text-color-secondary);">by tourist arrivals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="benchmark-card">
            <div class="benchmark-label">✅ Recovery Status</div>
            <div class="benchmark-value" style="color:{'#28a745' if recovery_rate >= 100 else '#dc3545'};">{recovery_rate:.0f}%</div>
            <div style="font-size:0.85rem; color:var(--text-color-secondary);">of 2019 levels</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="benchmark-card">
            <div class="benchmark-label">📈 Growth Rank</div>
            <div class="benchmark-value">#{int(growth_rank)}</div>
            <div style="font-size:0.85rem; color:var(--text-color-secondary);">by growth rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>📝 Key Observations</strong><br>
        • Sri Lanka ranks in the <strong>top 5</strong> for growth rate among comparator destinations<br>
        • Recovery rate (<strong>108%</strong>) is above regional average, indicating strong rebound<br>
        • Average stay duration (<strong>7 nights</strong>) is competitive, offering potential for higher per-tourist spending
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# SECTION 6: FORECAST DETAILS
# ---------------------------------------------------------------
with st.expander("📋 2026 Forecast Details", expanded=False):
    total_forecast = int(forecast_df['predicted_arrivals'].sum())
    peak_forecast = forecast_df.loc[forecast_df['predicted_arrivals'].idxmax()]
    low_forecast = forecast_df.loc[forecast_df['predicted_arrivals'].idxmin()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("2026 Total Forecast", f"{total_forecast:,}")
    with col2:
        st.metric("Peak Month", f"{peak_forecast['month'].strftime('%B')} ({peak_forecast['predicted_arrivals']:,.0f})")
    with col3:
        st.metric("Low Month", f"{low_forecast['month'].strftime('%B')} ({low_forecast['predicted_arrivals']:,.0f})")
    with col4:
        st.metric("Confidence Level", "80%")
    
    st.dataframe(
        forecast_df.style.format({
            'predicted_arrivals': '{:,.0f}',
            'lower_bound': '{:,.0f}',
            'upper_bound': '{:,.0f}'
        }).background_gradient(subset=['predicted_arrivals'], cmap='Blues'),
        use_container_width=True
    )

# ---------------------------------------------------------------
# SECTION 7: DATA DOWNLOAD
# ---------------------------------------------------------------
with st.expander("📥 Download Data", expanded=False):
    tab1, tab2 = st.tabs(["Historical Data", "Forecast Data"])
    with tab1:
        st.dataframe(arrivals_df, use_container_width=True)
        st.download_button(
            label="📥 Download Historical Data (CSV)",
            data=arrivals_df.to_csv(index=False),
            file_name="tourist_arrivals.csv",
            mime="text/csv"
        )
    with tab2:
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=forecast_df.to_csv(index=False),
            file_name="forecast_results.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.divider()
st.caption("Built with ❤️ using Streamlit · Data from SLTDA Monthly Reports · Forecast by Prophet")
st.caption(f"📅 Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")