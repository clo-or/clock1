import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Time Series Forecaster", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM CSS INJECTION ---
st.markdown("""
<style>
/* Base Dark Theme Overrides */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    background-image: radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                      radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.15) 0%, transparent 50%);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background-color: rgba(30, 41, 59, 0.6) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1rem;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Main Header Gradient */
.title-gradient {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(to right, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle {
    color: #94a3b8;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# --- FORECASTING FUNCTION ---
@st.cache_data
def generate_forecast(df, horizon):
    date_col = None
    val_col = None
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                date_col = col
                break
            except:
                pass
                
    if date_col is None:
        date_col = df.columns[0]
        val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    else:
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                val_col = col
                break
        if val_col is None:
            val_col = df.columns[1] if df.columns[1] != date_col else df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col]).reset_index(drop=True)
    
    dates = df[date_col]
    y = df[val_col].values
    n_samples = len(y)
    
    if n_samples < 5:
        return None, "Not enough data points to forecast. Need at least 5."
        
    test_size = max(int(n_samples * 0.2), min(3, int(n_samples*0.1)))
    test_size = min(test_size, n_samples - 2)
    
    train_y = y[:-test_size]
    test_y = y[-test_size:]
    
    try:
        eval_model = ExponentialSmoothing(train_y, trend='add', seasonal=None, initialization_method="estimated")
        eval_fitted = eval_model.fit()
        pred_y = eval_fitted.forecast(test_size)
    except Exception as e:
        pred_y = np.repeat(train_y[-1], test_size)

    mae = mean_absolute_error(test_y, pred_y)
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    mape = mean_absolute_percentage_error(test_y, pred_y)

    try:
        full_model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
        full_fitted = full_model.fit()
        future_forecast = full_fitted.forecast(horizon)
    except:
        future_forecast = np.repeat(y[-1], horizon)

    freq = pd.infer_freq(dates)
    if freq is None:
        avg_diff = (dates.iloc[-1] - dates.iloc[0]) / (len(dates) - 1)
        future_dates = [dates.iloc[-1] + i * avg_diff for i in range(1, horizon + 1)]
    else:
        future_dates = pd.date_range(start=dates.iloc[-1], periods=horizon+1, freq=freq)[1:]

    return {
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
        "hist_dates": dates,
        "hist_values": y,
        "future_dates": pd.Series(future_dates),
        "future_values": future_forecast
    }, None


# --- UI LAYOUT ---
st.markdown('<div class="title-gradient">Predictive Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated Time Series Analysis & Forecasting Engine</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration Settings")
    st.markdown("Upload your univariate time series dataset (CSV) to get started.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.divider()
    
    st.subheader("Forecast Parameters")
    horizon = st.slider("Forecast Horizon (시평)", min_value=1, max_value=100, value=12, help="Number of periods to forecast into the future.")

if uploaded_file is not None:
    st.success(f"Successfully loaded {uploaded_file.name}")
    df = pd.read_csv(uploaded_file)
    
    with st.spinner("Analyzing and generating forecast..."):
        result, error = generate_forecast(df, horizon)
        
    if error:
        st.error(error)
    else:
        col1, col2, col3 = st.columns(3)
        metrics = result['metrics']
        
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">MAE (Mean Absolute Error)</div>
                <div class="metric-value">{metrics['mae']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">RMSE (Root Mean Square Error)</div>
                <div class="metric-value">{metrics['rmse']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">MAPE (Mean Absolute % Error)</div>
                <div class="metric-value">{(metrics['mape']*100):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📈 Forecast Visualization")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result['hist_dates'], 
            y=result['hist_values'],
            mode='lines',
            name='Historical Data',
            line=dict(color='#3b82f6', width=2)
        ))
        
        conn_x = [result['hist_dates'].iloc[-1], result['future_dates'].iloc[0]]
        conn_y = [result['hist_values'][-1], result['future_values'][0]]
        fig.add_trace(go.Scatter(
            x=conn_x, y=conn_y,
            mode='lines',
            showlegend=False,
            line=dict(color='#a78bfa', width=3, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=result['future_dates'], 
            y=result['future_values'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#a78bfa', width=3, dash='dash'),
            marker=dict(size=6, color='#c084fc')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Date"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Value"),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(30, 41, 59, 0.8)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
    st.markdown("### 📈 Example Dashboard Preview")
    # A placeholder metric to show the beautiful design even empty
    st.markdown('''
        <div style="display:flex; justify-content:center; align-items:center; height:30vh; opacity:0.5; border: 2px dashed rgba(255,255,255,0.2); border-radius:16px;">
            <p>Upload a file to see the interactive forecast visualization</p>
        </div>
    ''', unsafe_allow_html=True)
