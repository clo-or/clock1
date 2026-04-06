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
    page_title="지능형 시계열 예측", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT/PREMIUM CSS INJECTION ---
st.markdown("""
<style>
/* 밝은 테마 배경 및 텍스트 색상 오버라이드 */
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
    background-image: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                      radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0,0,0,0.05);
}

/* 글래스모피즘 카드 스타일 */
.glass-card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.05);
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1rem;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px 0 rgba(0, 0, 0, 0.1);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #2563eb;
    margin: 0.5rem 0;
}

.metric-label {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 700;
}

/* 메인 타이틀 그라데이션 */
.title-gradient {
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(to right, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle {
    color: #475569;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* 기본 텍스트 색상 조정 (다크모드 강제 해제 느낌) */
h1, h2, h3, p, span, div {
    color: #1e293b;
}

/* Sidebar 내부 텍스트 처리 */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #1e293b !important;
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
        return None, "데이터가 너무 적습니다. 최소 5개의 데이터 포인트가 필요합니다."
        
    test_size = max(int(n_samples * 0.2), min(3, int(n_samples*0.1)))
    test_size = min(test_size, n_samples - 2)
    
    train_y = y[:-test_size]
    test_y = y[-test_size:]
    
    try:
        eval_model = ExponentialSmoothing(train_y, trend='add', seasonal=None, initialization_method="estimated")
        eval_fitted = eval_model.fit()
        pred_y = eval_fitted.forecast(test_size)
    except Exception:
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

def get_sample_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.linspace(10, 50, 100) + np.sin(np.linspace(0, 20, 100)) * 10 + np.random.normal(0, 2, 100)
    return pd.DataFrame({"날짜": dates, "판매량": values})

# --- UI LAYOUT ---
st.markdown('<div class="title-gradient">지능형 시계열 예측 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">자동화된 데이터 분석 및 미래 예측 대시보드</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 설정 및 데이터 업로드")
    st.markdown("분석할 단변량 시계열 데이터(CSV)를 업로드해주세요.")
    
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    st.markdown("또는 아래 버튼을 눌러 샘플 데이터를 사용해보세요.")
    use_sample = st.button("📊 샘플 데이터로 실행해보기", use_container_width=True)
    
    st.divider()
    
    st.subheader("예측 파라미터")
    horizon = st.slider("예측 구간 (시평)", min_value=1, max_value=100, value=12, help="미래에 예측할 기간의 수를 설정합니다.")

# 의사결정: 업로드 파일이 있거나 샘플 버튼을 눌렀을 경우
st.session_state['use_sample'] = st.session_state.get('use_sample', False)
if use_sample:
    st.session_state['use_sample'] = True
if uploaded_file is not None:
    st.session_state['use_sample'] = False # 업로드하면 샘플모드 해제

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.")
elif st.session_state['use_sample']:
    df = get_sample_data()
    st.info("💡 샘플 데이터를 사용하여 데모를 실행 중입니다. (임의의 가상 판매량 데이터)")

if df is not None:
    with st.spinner("데이터 분석 및 예측 모델 비딩 중..."):
        result, error = generate_forecast(df, horizon)
        
    if error:
        st.error(error)
    else:
        col1, col2, col3 = st.columns(3)
        metrics = result['metrics']
        
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">MAE (평균 절대 오차)</div>
                <div class="metric-value">{metrics['mae']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">RMSE (평균 제곱근 오차)</div>
                <div class="metric-value">{metrics['rmse']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">MAPE (평균 절대 비율 오차)</div>
                <div class="metric-value">{(metrics['mape']*100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📈 시계열 데이터 및 예측 시각화")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result['hist_dates'], 
            y=result['hist_values'],
            mode='lines',
            name='과거 실제 데이터',
            line=dict(color='#2563eb', width=2)
        ))
        
        conn_x = [result['hist_dates'].iloc[-1], result['future_dates'].iloc[0]]
        conn_y = [result['hist_values'][-1], result['future_values'][0]]
        fig.add_trace(go.Scatter(
            x=conn_x, y=conn_y,
            mode='lines',
            showlegend=False,
            line=dict(color='#8b5cf6', width=3, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=result['future_dates'], 
            y=result['future_values'],
            mode='lines+markers',
            name='AI 예측 결과',
            line=dict(color='#8b5cf6', width=3, dash='dash'),
            marker=dict(size=6, color='#a78bfa')
        ))
        
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(255,255,255,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', title="날짜 (Date)"),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', title="값 (Value)"),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.info("👈 좌측 사이드바에서 CSV 파일을 업로드하시거나 '샘플 데이터로 실행해보기'를 클릭하세요.")
    st.markdown("### 📈 빈 대시보드 미리보기")
    st.markdown('''
        <div style="display:flex; justify-content:center; align-items:center; height:30vh; background: rgba(255,255,255,0.5); border: 2px dashed rgba(0,0,0,0.2); border-radius:16px;">
            <p style="color: #64748b; font-weight: 500;">파일을 업로드하면 이곳에 인터랙티브 예측 차트가 표시됩니다.</p>
        </div>
    ''', unsafe_allow_html=True)
