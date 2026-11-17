import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title='Stock Analyzer Pro', page_icon='📈', layout='wide')
st.title('📈 Stock Analyzer Pro')
st.caption('PER/PBR · Stochastic + MACD + ADX · 10-year historical backtest')

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker):
    obj = yf.Ticker(ticker)
    hist = obj.history(period='10y', auto_adjust=True)
    if hist.empty:
        raise ValueError('주가 데이터를 찾지 못했습니다. 티커를 확인하세요.')
    hist = hist[['Close', 'High', 'Low', 'Open', 'Volume']].dropna().copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    try:
        info = obj.info
    except Exception:
        info = {}
    return hist, info

def indicators(df):
    d = df.copy()
    low14 = d.Low.rolling(14, min_periods=14).min()
    high14 = d.High.rolling(14, min_periods=14).max()
    d['K'] = (100 * (d.Close - low14) / (high14 - low14).replace(0, np.nan)).rolling(3, min_periods=3).mean()
    d['D'] = d.K.rolling(3, min_periods=3).mean()
    ema12 = d.Close.ewm(span=12, adjust=False).mean()
    ema26 = d.Close.ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_signal'] = d.MACD.ewm(span=9, adjust=False).mean()
    prev = d.Close.shift(1)
    tr = pd.concat([(d.High-d.Low), (d.High-prev).abs(), (d.Low-prev).abs()], axis=1).max(axis=1)
    up, down = d.High.diff(), -d.Low.diff()
    plus = pd.Series(np.where((up>down)&(up>0), up, 0.0), index=d.index)
    minus = pd.Series(np.where((down>up)&(down>0), down, 0.0), index=d.index)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    pdi = 100*plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/atr.replace(0,np.nan)
    mdi = 100*minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['ADX'] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d['plusDI'],d['minusDI'] = pdi,mdi
    # Signals are known only after the close of the corresponding day.
    d['stoch_event'] = (d.K.shift(1)<=d.D.shift(1)) & (d.K>d.D) & (d.K<80)
    d['macd_event'] = (d.MACD.shift(1)<=d.MACD_signal.shift(1)) & (d.MACD>d.MACD_signal)
    d['adx_event'] = (d.ADX>=25) & (d.plusDI>d.minusDI) & (d.ADX>d.ADX.shift(1))
    return d

def strategy(d, validity=5, hold=20, fee_bps=10):
    # Rolling window includes the current signal date; orders fill at next day's open.
    a = d.stoch_event.rolling(validity, min_periods=1).max().fillna(0).astype(bool)
    b = d.macd_event.rolling(validity, min_periods=1).max().fillna(0).astype(bool)
    combined = a & b & d.adx_event.fillna(False)
    # One trade at a time; fixed holding horizon, next-open entry and exit.
    trades=[]
    equity = pd.Series(index=d.index, dtype=float)
    cash=1.0
    i=0
    while i<len(d)-hold-1:
        if combined.iloc[i] and pd.notna(d.Open.iloc[i+1]) and pd.notna(d.Open.iloc[i+1+hold]):
            entry_i=i+1
            exit_i=entry_i+hold
            buy=float(d.Open.iloc[entry_i]); sell=float(d.Open.iloc[exit_i])
            if buy<=0:
                i+=1; continue
            gross=sell/buy
            net=gross*(1-fee_bps/10000)**2
            trades.append({'신호일':d.index[i].date(), '매수일':d.index[entry_i].date(), '매도일':d.index[exit_i].date(), '매수가':buy, '매도가':sell, '수익률(%)':(net-1)*100})
            equity.iloc[i:entry_i]=cash
            for j in range(entry_i, exit_i):
                equity.iloc[j]=cash*(float(d.Close.iloc[j])/buy)*(1-fee_bps/10000)
            cash*=net
            equity.iloc[exit_i]=cash
            i=exit_i+1
        else:
            equity.iloc[i]=cash
            i+=1
    equity=equity.ffill().fillna(1.0)
    return pd.DataFrame(trades),equity,combined

def max_dd(series):
    if series.empty: return float('nan')
    return (series/series.cummax()-1).min()*100

with st.sidebar:
    st.header('분석 설정')
    ticker=st.text_input('티커 (예: AAPL, MSFT, 005930.KS)', 'AAPL').strip().upper()
    validity=st.number_input('신호 유효기간 (거래일)', 1, 20, 5)
    hold=st.number_input('보유기간 (거래일)', 1, 120, 20)
    fee=st.number_input('편도 거래비용 (bps)', 0, 100, 10, help='10bps = 0.1%, 매수·매도 각각 부과')
    st.caption('과거 데이터는 Yahoo Finance를 사용합니다. 가격은 자동 조정 종가이며, 매매는 다음 거래일 시가를 가정합니다.')

if not ticker:
    st.info('티커를 입력하세요.')
    st.stop()
try:
    with st.spinner('주가와 지표를 불러오는 중...'):
        hist, info=get_data(ticker)
        d=indicators(hist)
        trades, equity, signals=strategy(d, validity, hold, fee)
except Exception as exc:
    st.error(f'데이터 로딩 실패: {exc}')
    st.stop()

name=info.get('shortName') or ticker
st.subheader(f'{name} ({ticker})')
t1,t2,t3=st.tabs(['📊 주가·가치평가', '📉 기술적 분석', '🧪 백테스트'])
with t1:
    last=float(d.Close.iloc[-1])
    c1,c2,c3,c4=st.columns(4)
    c1.metric('최근 조정 종가', f'{last:,.2f}')
    pe=info.get('trailingPE'); pb=info.get('priceToBook')
    c2.metric('현재 PER', f'{pe:.2f}배' if isinstance(pe,(int,float)) and np.isfinite(pe) else '자료 없음')
    c3.metric('현재 PBR', f'{pb:.2f}배' if isinstance(pb,(int,float)) and np.isfinite(pb) else '자료 없음')
    c4.metric('주가 데이터', f'{len(d):,}일')
    fig=go.Figure(go.Scatter(x=d.index,y=d.Close,name='조정 종가'))
    fig.update_layout(title='최대 10년 주가', xaxis_title='날짜', yaxis_title='가격', height=440)
    st.plotly_chart(fig, use_container_width=True)
    st.warning('10년 PER/PBR 밴드는 아직 제공하지 않습니다. Yahoo Finance의 현재 PER/PBR을 과거 전체에 적용하면 잘못된 밴드가 만들어집니다. 과거 시점별 EPS/BPS 재무 데이터 소스를 연결한 후 추가할 예정입니다.')
with t2:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.index,y=d.K,name='%K'))
    fig.add_trace(go.Scatter(x=d.index,y=d.D,name='%D'))
    fig.add_hline(y=80,line_dash='dot');fig.add_hline(y=20,line_dash='dot')
    fig.update_layout(title='스토캐스틱 (14,3,3)',height=320)
    st.plotly_chart(fig,use_container_width=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.index,y=d.MACD,name='MACD'))
    fig.add_trace(go.Scatter(x=d.index,y=d.MACD_signal,name='Signal'))
    fig.update_layout(title='MACD (12,26,9)',height=320)
    st.plotly_chart(fig,use_container_width=True)
    fig=go.Figure()
    for k in ['ADX','plusDI','minusDI']:
        fig.add_trace(go.Scatter(x=d.index,y=d[k],name=k))
    fig.add_hline(y=25,line_dash='dot')
    fig.update_layout(title='ADX / +DI / -DI (14)',height=320)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('3중 조건: 최근 N거래일 내 스토캐스틱 상향교차(K<80)와 MACD 상향교차, 당일 ADX≥25 및 +DI>-DI 및 ADX 상승.')
    st.metric('최근 거래일 3중 조건', '충족' if bool(signals.iloc[-1]) else '미충족')
with t3:
    n=len(trades)
    wins=float((trades['수익률(%)']>0).mean()*100) if n else 0.0
    avg=float(trades['수익률(%)'].mean()) if n else 0.0
    # Benchmark uses the same start and end as the entire backtest window.
    bench=(d.Close/d.Close.iloc[0]).ffill()
    c1,c2,c3,c4=st.columns(4)
    c1.metric('거래 횟수',str(n))
    c2.metric('승률',f'{wins:.1f}%' if n else '거래 없음')
    c3.metric('평균 거래 수익률',f'{avg:+.2f}%' if n else '거래 없음')
    c4.metric('전략 MDD',f'{max_dd(equity):.2f}%')
    st.write(f'전략 누적 수익률: **{(equity.iloc[-1]-1)*100:+.2f}%** | 동일 종목 매수보유: **{(bench.iloc[-1]-1)*100:+.2f}%** | 매수보유 MDD: **{max_dd(bench):.2f}%**')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=equity.index,y=equity,name='전략 (초기자산=1)'))
    fig.add_trace(go.Scatter(x=bench.index,y=bench,name='동일 종목 매수보유'))
    fig.update_layout(title='자산곡선 비교',height=440,yaxis_title='초기자산 대비 배수')
    st.plotly_chart(fig,use_container_width=True)
    if n:
        st.dataframe(trades.iloc[::-1],use_container_width=True,hide_index=True)
        st.download_button('거래 내역 CSV',trades.to_csv(index=False).encode('utf-8-sig'),file_name=f'{ticker}_trades.csv',mime='text/csv')
    else:
        st.info('해당 기간에 조건을 충족한 거래가 없습니다.')
    st.caption('교육·연구용 백테스트입니다. 미래 수익률을 보장하지 않습니다. 슬리피지, 세금, 배당 현금흐름, 유동성, 생존편향 등을 완전히 반영하지 않습니다. 조정 OHLC는 실제 체결가와 다를 수 있습니다.')
