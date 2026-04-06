import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

try:
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing as SktimeExponentialSmoothing
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import mean_absolute_error as sktime_mae
    from sktime.performance_metrics.forecasting import mean_squared_error as sktime_mse
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as sktime_mape
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    
st.set_page_config(page_title="종합 시계열 분석 엔진", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
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
.metric-value { font-size: 2.2rem; font-weight: 800; color: #2563eb; margin: 0.5rem 0; }
.metric-label { color: #64748b; font-size: 0.9rem; font-weight: 700; }
.title-gradient {
    font-size: 2.5rem; font-weight: 900;
    background: linear-gradient(to right, #2563eb, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
}
.subtitle { color: #475569; font-size: 1.1rem; margin-bottom: 2rem; }
h1, h2, h3, p, span, div { color: #1e293b; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #1e293b !important; }
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 0px; background: transparent; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_sample_data():
    import os
    target_csv = "대전광역시 서구_관저문예회관 홈페이지 방문자 현황_20260324.csv"
    if os.path.exists(target_csv):
        df = pd.read_csv(target_csv, encoding='cp949')
        # 샘플 데이터에 전처리 시연을 위한 가상의 결측치 추가
        if len(df) > 20:
            df.loc[10:13, df.columns[2]] = np.nan
        return df
    else:
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        values = np.linspace(10, 50, 100) + np.sin(np.linspace(0, 20, 100)) * 10 + np.random.normal(0, 2, 100)
        df = pd.DataFrame({"날짜": dates, "판매량": values})
        df.loc[20:25, "판매량"] = np.nan
        return df

# --- UI LAYOUT ---
st.markdown('<div class="title-gradient">지능형 시계열 예측 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">데이터 전처리, 계절성 분해 및 sktime 머신러닝 예측 시스템</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 데이터 로드")
    uploaded_file = st.file_uploader("단변량 시계열 CSV 업로드", type=["csv"])
    use_sample = st.button("📊 샘플 데이터 사용", use_container_width=True)
    
    st.divider()
    st.subheader("예측 파라미터")
    horizon = st.slider("미래 예측 구간 (시평)", min_value=1, max_value=100, value=12, help="미래에 예측할 기간의 수를 설정합니다.")
    model_choice = st.selectbox("예측 모델 (sktime/statsmodels)", ["Holt-Winters 평활법", "Naive (단순 이동)"])

    st.markdown("<div style='margin-top: 30px; text-align: center; color: #94a3b8; font-size: 0.95em; font-weight: 600;'>C321050 이승아<br>스마트제조 프로젝트 1</div>", unsafe_allow_html=True)

st.session_state['use_sample'] = st.session_state.get('use_sample', False)
if use_sample:
    st.session_state['use_sample'] = True
if uploaded_file is not None:
    st.session_state['use_sample'] = False

df_raw = None
if uploaded_file is not None:
    try:
        try:
            df_raw = pd.read_csv(uploaded_file, thousands=',')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, thousands=',', encoding='cp949')
        st.success(f"'{uploaded_file.name}' 업로드 성공.")
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
elif st.session_state['use_sample']:
    df_raw = get_sample_data()
    st.info("💡 샘플 데이터 명칭: [대전광역시 서구_관저문예회관 홈페이지 방문자 현황](https://www.data.go.kr/data/15039305/fileData.do)")
if df_raw is not None:
    # 가로형 시계열 데이터 자동 전치 (행은 적은데 열이 많은 경우)
    if len(df_raw) <= 5 and len(df_raw.columns) > 5:
        try:
            df_raw = df_raw.set_index(df_raw.columns[0]).T.reset_index()
            df_raw.rename(columns={'index': '기간'}, inplace=True)
            st.info("💡 데이터가 가로로 긴 형태여서, 시계열 분석을 위해 자동으로 세로 방향(전치)으로 변환했습니다.")
        except:
            pass

if df_raw is not None and len(df_raw) > 5:
    st.markdown("### 🗂️ 데이터 컬럼 매핑 (전처리)")
    cols = df_raw.columns.tolist()
    
    # 자동 추천 로직
    def_date = next((c for c in cols if '날짜' in c or 'date' in c.lower() or '일자' in c), cols[0])
    def_val = next((c for c in reversed(cols) if c != def_date and pd.api.types.is_numeric_dtype(df_raw[c])), cols[-1])
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        date_col = st.selectbox("시간/날짜 컬럼 선택", cols, index=cols.index(def_date) if def_date in cols else 0)
    with col_sel2:
        val_col = st.selectbox("분석/예측 대상 컬럼 선택", cols, index=cols.index(def_val) if def_val in cols else len(cols)-1)
    
    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df_raw = df_raw.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df_raw[val_col] = pd.to_numeric(df_raw[val_col], errors='coerce')
    
    st.divider()
    tab1, tab2, tab3 = st.tabs(["🧹 데이터 전처리", "🔍 시계열 분해", "🚀 모델 예측"])
    
    # ---------------- TAB 1: PREPROCESSING ----------------
    with tab1:
        st.subheader("데이터 결측치 보간 및 정상성 변환")
        
        c1, c2 = st.columns(2)
        with c1:
            interpolate_method = st.selectbox("결측치 보간법 선택", ["선형 보간 (Linear)", "스플라인 보간 (Spline)", "삭제 (Drop)"])
        with c2:
            transform_method = st.selectbox("정상성(Stationarity) 변환", ["적용 안 함", "로그 변환 (Log Transform)", "1차 차분 (1st Differencing)"])
            
        df_processed = df_raw.copy()
        
        if df_processed[val_col].isnull().any():
            st.warning(f"⚠️ 원본 데이터에 **{df_processed[val_col].isnull().sum()}개**의 결측치가 존재합니다.")
            
        if interpolate_method == "선형 보간 (Linear)":
            df_processed[val_col] = df_processed[val_col].interpolate(method='linear')
        elif interpolate_method == "스플라인 보간 (Spline)":
            try:
                df_processed[val_col] = df_processed[val_col].interpolate(method='cubicspline')
            except:
                df_processed[val_col] = df_processed[val_col].interpolate(method='linear') # Fallback if spline fails
        else:
            df_processed = df_processed.dropna(subset=[val_col]).reset_index(drop=True)
            
        # 남은 결측치 제거
        df_processed = df_processed.dropna(subset=[val_col]).reset_index(drop=True)
        
        if len(df_processed) > 5:
            if transform_method == "로그 변환 (Log Transform)":
                min_val = df_processed[val_col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                df_processed[val_col] = np.log((df_processed[val_col] + offset).astype(float))
            elif transform_method == "1차 차분 (1st Differencing)":
                df_processed[val_col] = df_processed[val_col].diff()
                df_processed = df_processed.dropna(subset=[val_col]).reset_index(drop=True)

            fig_pre = go.Figure()
            fig_pre.add_trace(go.Scatter(x=df_raw[date_col], y=df_raw[val_col], mode='lines', line=dict(color='rgba(255, 0, 0, 0.4)', width=5), name="원본"))
            fig_pre.add_trace(go.Scatter(x=df_processed[date_col], y=df_processed[val_col], mode='lines', line=dict(color='#2563eb'), name="전처리 완료"))
            fig_pre.update_layout(template='plotly_white', title="원본 vs 전처리 비교", hovermode='x unified')
            st.plotly_chart(fig_pre, use_container_width=True)
        else:
            st.error("유효한 데이터가 너무 부족합니다.")

    # ---------------- TAB 2: DECOMPOSITION ----------------
    with tab2:
        st.subheader("계절성 및 추세 성분 분해 (STL / Classical)")
        period = st.number_input("계절성 주기", min_value=2, max_value=365, value=7)
        
        if len(df_processed) > period * 2:
            try:
                ts = df_processed[val_col].values
                decomposition = seasonal_decompose(ts, model='additive', period=int(period))
                
                fig_dec = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                        subplot_titles=("1. 원본", "2. 추세 (Trend)", "3. 계절성 (Seasonality)", "4. 잔차 (Residuals)"))
                x_vals = df_processed[date_col]
                
                fig_dec.add_trace(go.Scatter(x=x_vals, y=ts, line=dict(color='#2563eb')), row=1, col=1)
                fig_dec.add_trace(go.Scatter(x=x_vals, y=decomposition.trend, line=dict(color='#ef4444')), row=2, col=1)
                fig_dec.add_trace(go.Scatter(x=x_vals, y=decomposition.seasonal, line=dict(color='#10b981')), row=3, col=1)
                fig_dec.add_trace(go.Scatter(x=x_vals, y=decomposition.resid, mode='markers', marker=dict(color='#8b5cf6', size=4)), row=4, col=1)
                
                fig_dec.update_layout(height=800, template='plotly_white', showlegend=False)
                st.plotly_chart(fig_dec, use_container_width=True)
            except Exception as e:
                st.error(f"분해 중 오류가 발생했습니다: {e}")
        else:
            st.warning("데이터가 주기를 시각화하기에 충분하지 않습니다.")

    # ---------------- TAB 3: MODELING ----------------
    with tab3:
        st.subheader(f"시계열 성능 검증 및 미래 예측")
        if not SK_AVAILABLE:
            st.warning("⚠️ `sktime` 모듈 오류로 인해 `statsmodels` 폴백 모드를 사용 중입니다.")
            
        y_raw = df_processed[val_col].values
        dates = df_processed[date_col].reset_index(drop=True)
        
        if len(y_raw) > 10:
            test_size = max(int(len(y_raw) * 0.2), 3)

            if SK_AVAILABLE:
                y = pd.Series(y_raw)
                y.index = pd.RangeIndex(len(y))
                
                train_y, test_y = temporal_train_test_split(y, test_size=test_size)
                
                if model_choice == "Holt-Winters 평활법":
                    model = SktimeExponentialSmoothing(trend='add', seasonal=None)
                else:
                    model = NaiveForecaster(strategy="last")
                
                try:
                    model.fit(train_y)
                    fh_test = np.arange(1, len(test_y) + 1)
                    pred_y = model.predict(fh=fh_test)
                    
                    mae = sktime_mae(test_y, pred_y)
                    rmse = np.sqrt(sktime_mse(test_y, pred_y))
                    mape = sktime_mape(test_y, pred_y)
                    
                    model_full = SktimeExponentialSmoothing(trend='add', seasonal=None) if model_choice == "Holt-Winters 평활법" else NaiveForecaster(strategy="last")
                    model_full.fit(y)
                    future_forecast = model_full.predict(fh=np.arange(1, horizon + 1))
                except Exception as e:
                    st.error(f"모델 연산 중 오류 발생 (단순 이동 평균으로 대체합니다): {e}")
                    fallback_val = float(train_y.iloc[-1]) if len(train_y) > 0 else 0.0
                    pred_y = pd.Series(np.repeat(fallback_val, len(test_y)), index=test_y.index)
                    future_forecast = pd.Series(np.repeat(fallback_val, horizon))
                    mae, rmse, mape = 0, 0, 0
            else:
                train_y = y_raw[:-test_size]
                test_y = y_raw[-test_size:]
                try:
                    if model_choice == "Naive (단순 이동)":
                        pred_y = np.repeat(train_y[-1], test_size)
                        future_forecast = np.repeat(y_raw[-1], horizon)
                    else:
                        eval_model = ExponentialSmoothing(train_y, trend='add', seasonal=None, initialization_method="estimated")
                        pred_y = eval_model.fit(optimized=True).forecast(test_size)
                        full_model = ExponentialSmoothing(y_raw, trend='add', seasonal=None, initialization_method="estimated")
                        future_forecast = full_model.fit(optimized=True).forecast(horizon)
                    
                    mae = mean_absolute_error(test_y, pred_y)
                    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
                    mape = mean_absolute_percentage_error(test_y, pred_y)
                except Exception as e:
                    st.error(f"모델 연산 중 오류 발생 (대체값 출력): {e}")
                    fallback_val = float(train_y[-1]) if len(train_y) > 0 else 0.0
                    pred_y = np.repeat(fallback_val, test_size)
                    future_forecast = np.repeat(fallback_val, horizon)
                    mae, rmse, mape = 0, 0, 0

            # UI rendering
            st.markdown(f"**Train (80%) / Test (20%) 정확성 검증결과**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="glass-card"><div class="metric-label">MAE</div><div class="metric-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="glass-card"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.2f}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="glass-card"><div class="metric-label">MAPE</div><div class="metric-value">{(mape * 100):.1f}%</div></div>', unsafe_allow_html=True)
                
            freq = pd.infer_freq(dates)
            if freq is None:
                avg_diff = (dates.iloc[-1] - dates.iloc[0]) / (len(dates) - 1)
                future_dates = [dates.iloc[-1] + i * avg_diff for i in range(1, horizon + 1)]
            else:
                future_dates = pd.date_range(start=dates.iloc[-1], periods=horizon+1, freq=freq)[1:]

            fig = go.Figure()
            
            # Make sure everything is array
            train_y_vals = np.array(train_y)
            test_y_vals = np.array(test_y)
            pred_y_vals = np.array(pred_y)
            future_forecast_vals = np.array(future_forecast)
            
            fig.add_trace(go.Scatter(x=dates.iloc[:-test_size], y=train_y_vals, mode='lines', name='Train 데이터', line=dict(color='#cbd5e1')))
            fig.add_trace(go.Scatter(x=dates.iloc[-test_size:], y=test_y_vals, mode='lines', name='Test 데이터 (실제)', line=dict(color='#3b82f6', width=2)))
            fig.add_trace(go.Scatter(x=dates.iloc[-test_size:], y=pred_y_vals, mode='lines', name='Test 예측', line=dict(color='#ef4444', dash='dot', width=3)))
            
            if len(future_dates) > 0 and len(future_forecast_vals) > 0:
                conn_x = [dates.iloc[-1], future_dates[0]]
                conn_y = [y_raw[-1], future_forecast_vals[0]]
                fig.add_trace(go.Scatter(x=conn_x, y=conn_y, mode='lines', showlegend=False, line=dict(color='#a855f7', width=3, dash='dash')))
                fig.add_trace(go.Scatter(x=future_dates, y=future_forecast_vals, mode='lines+markers', name='미래 시평 예측', line=dict(color='#a855f7', width=3, dash='dash'), marker=dict(size=6)))

            fig.update_layout(template='plotly_white', hovermode='x unified', title="데이터 검증 및 미래 예측 시각화")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("학습에 필요한 최소 데이터 포인트가 부족합니다.")
else:
    if df_raw is not None and len(df_raw) <= 5:
        st.error(f"데이터의 세로 길이(Data points)가 너무 짧습니다 (현재 {len(df_raw)}개). 일반적인 형태의 시계열 데이터인지 확인해주세요.")
    elif uploaded_file is None and not use_sample:
        st.info("👈 사이드바에서 데이터를 업로드하여 시작해보세요.")
