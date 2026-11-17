import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scanner import render_scanner

st.set_page_config(page_title='MyStockQuant', page_icon='📈', layout='wide')
st.title('📈 MyStockQuant')
st.caption('주식 가치평가 · 3중 기술적 신호 · 최대 10년 백테스트')

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker):
    obj = yf.Ticker(ticker)
    hist = obj.history(period='10y', auto_adjust=True, actions=False)
    if hist.empty:
        raise ValueError('주가 데이터가 없습니다. 티커를 확인하거나 잠시 후 다시 시도하세요.')
    hist = hist[['Open','High','Low','Close','Volume']].dropna().copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    try:
        info = obj.info or {}
    except Exception:
        info = {}
    return hist, info

def indicators(df, oversold=20, adx_level=25):
    d = df.copy()
    lo = d.Low.rolling(14).min()
    hi = d.High.rolling(14).max()
    raw = 100 * (d.Close - lo) / (hi - lo).replace(0, np.nan)
    d['K'] = raw.rolling(3).mean()
    d['D'] = d.K.rolling(3).mean()
    e12 = d.Close.ewm(span=12, adjust=False).mean()
    e26 = d.Close.ewm(span=26, adjust=False).mean()
    d['MACD'] = e12-e26
    d['MACD_signal'] = d.MACD.ewm(span=9, adjust=False).mean()
    prev = d.Close.shift()
    tr = pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    up, down = d.High.diff(), -d.Low.diff()
    plus = pd.Series(np.where((up>down)&(up>0),up,0.0),index=d.index)
    minus = pd.Series(np.where((down>up)&(down>0),down,0.0),index=d.index)
    atr = tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    pdi = 100*plus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.replace(0,np.nan)
    mdi = 100*minus.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['ADX'] = dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d['plusDI'], d['minusDI'] = pdi,mdi
    cross_stoch = (d.K.shift(1)<=d.D.shift(1)) & (d.K>d.D)
    d['stoch_event'] = (cross_stoch & (pd.concat([d.K.shift(1),d.D.shift(1)],axis=1).min(axis=1)<=oversold)).fillna(False)
    d['macd_event'] = ((d.MACD.shift(1)<=d.MACD_signal.shift(1)) & (d.MACD>d.MACD_signal)).fillna(False)
    d['adx_event'] = ((d.ADX>=adx_level) & (d.plusDI>d.minusDI) & (d.ADX>d.ADX.shift(1))).fillna(False)
    d['SMA200'] = d.Close.rolling(200).mean()
    return d

def get_signals(d, validity):
    s = d.stoch_event.rolling(validity,min_periods=1).max().astype(bool)
    m = d.macd_event.rolling(validity,min_periods=1).max().astype(bool)
    a = d.adx_event.astype(bool)
    return s,m,a,s&m&a

def run_backtest(d, combined, hold, fee_bps):
    # Signals are observed after closing. Trades enter/exit at the next available open.
    # One position at a time; mark to market at each close. Cash stays flat when not invested.
    n=len(d)
    curve=np.ones(n,dtype=float)
    trades=[]
    capital=1.0
    i=0
    while i<n:
        if i+hold+1<n and bool(combined.iloc[i]):
            entry=i+1
            exit_idx=entry+hold
            buy=float(d.Open.iloc[entry]); sell=float(d.Open.iloc[exit_idx])
            if not (np.isfinite(buy) and np.isfinite(sell) and buy>0 and sell>0):
                curve[i]=capital
                i+=1
                continue
            cost=fee_bps/10000.0
            curve[i]=capital
            units=capital*(1-cost)/buy
            # Values are measured at each day's close, including entry day.
            for j in range(entry,exit_idx):
                curve[j]=units*float(d.Close.iloc[j])
            capital=units*sell*(1-cost)
            curve[exit_idx]=capital
            net_return=(sell/buy)*(1-cost)**2-1
            trades.append({'신호일':d.index[i].date(), '매수일':d.index[entry].date(),
                           '매도일':d.index[exit_idx].date(), '매수가':round(buy,4),
                           '매도가':round(sell,4), '수익률(%)':round(net_return*100,3),
                           '장세': '상승 추세' if pd.notna(d.SMA200.iloc[i]) and d.Close.iloc[i]>d.SMA200.iloc[i] else ('하락 추세' if pd.notna(d.SMA200.iloc[i]) else '판별 불가')})
            i=exit_idx+1
        else:
            curve[i]=capital
            i+=1
    return pd.DataFrame(trades),pd.Series(curve,index=d.index,name='전략')

def mdd(x):
    return float((x/x.cummax()-1).min()*100)

def fmt(x):
    return f'{float(x):.2f}' if pd.notna(x) else '자료 없음'

def last_event_date(series):
    dates=series.index[series.fillna(False)]
    return str(dates[-1].date()) if len(dates) else '발생 기록 없음'

with st.sidebar:
    st.header('분석 설정')
    ticker=st.text_input('티커 (예: AAPL, MSFT, 005930.KS)','MSFT').strip().upper()
    validity=st.number_input('신호 유효기간 (거래일)',min_value=1,max_value=20,value=5)
    hold=st.number_input('보유기간 (거래일)',min_value=1,max_value=120,value=20)
    fee=st.number_input('편도 거래비용 (bps)',min_value=0,max_value=100,value=10,help='10bps = 0.1%; 매수와 매도 각각 반영')
    oversold=st.slider('스토캐스틱 과매도 기준',5,50,20,5)
    adx_level=st.slider('ADX 최소 기준',15,40,25,1)
    chart_days=st.selectbox('기술 차트 표시 구간',['최근 1년','최근 3년','최대 10년'],index=0)
    st.caption('Yahoo Finance 데이터를 사용합니다. 요청 제한 시 잠시 후 다시 시도하세요. 지표는 일봉 종가 확정 후 계산합니다.')
if not ticker:
    st.info('티커를 입력하세요.');st.stop()
try:
    with st.spinner('데이터 분석 중...'):
        hist,info=get_data(ticker)
        d=indicators(hist,oversold,adx_level)
        ss,ms,ads,combined=get_signals(d,validity)
        trades,equity=run_backtest(d,combined,hold,fee)
except Exception as exc:
    st.error(f'데이터 로딩 실패: {exc}')
    st.info('무료 데이터 공급원의 요청 제한일 수 있습니다. 잠시 후 새로고침하거나 다른 종목을 확인해 보세요.')
    st.stop()

st.subheader(f"{info.get('shortName') or ticker} ({ticker})")
st.caption(f'마지막 데이터 날짜: {d.index[-1].date()} · 실제 거래 권유가 아닌 연구용 신호')
last=d.iloc[-1]
labels=[('스토캐스틱',bool(ss.iloc[-1]),f"%K {fmt(last.K)} / %D {fmt(last.D)}",f'최근 {validity}거래일 내 과매도 상향교차',last_event_date(d.stoch_event)),
        ('MACD',bool(ms.iloc[-1]),f"MACD {fmt(last.MACD)} / Signal {fmt(last.MACD_signal)}",f'최근 {validity}거래일 내 상향교차',last_event_date(d.macd_event)),
        ('ADX',bool(ads.iloc[-1]),f"ADX {fmt(last.ADX)} / +DI {fmt(last.plusDI)} / -DI {fmt(last.minusDI)}",f'ADX ≥ {adx_level}, 상승 중, +DI > -DI',last_event_date(d.adx_event))]

t1,t2,t3,t4=st.tabs(['🧭 신호 대시보드','📉 기술적 분석','🧪 백테스트','🏆 한미 종목 랭킹'])
with t1:
    st.subheader('현재 3중 매수 후보 상태')
    cols=st.columns(3)
    for col,(name,ok,values,rule,event_date) in zip(cols,labels):
        with col:
            st.metric(name,'✅ 조건 충족' if ok else '⚪ 미충족')
            st.caption(values)
            st.write(rule)
            st.caption(f'마지막 개별 신호: {event_date}')
    if bool(combined.iloc[-1]):
        st.success('3중 조건 충족: 매수 후보 신호입니다. 수익이 보장되는 것은 아닙니다.')
    else:
        st.info('3중 조건 미충족: 현재 설정의 신규 매수 후보 신호가 없습니다.')
    st.caption('스토캐스틱·MACD 이벤트는 최근 N거래일 동안 유효하며, ADX 조건은 현재 거래일에 충족해야 합니다. 각 조건은 독립적으로 계산됩니다.')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.index,y=d.Close,name='조정 종가',line=dict(color='#2678d4')))
    if len(trades):
        ent=pd.to_datetime(trades['매수일']); ex=pd.to_datetime(trades['매도일'])
        fig.add_trace(go.Scatter(x=ent,y=d.loc[ent,'Open'],mode='markers',name='매수',marker=dict(symbol='triangle-up',size=11,color='#159f75')))
        fig.add_trace(go.Scatter(x=ex,y=d.loc[ex,'Open'],mode='markers',name='매도',marker=dict(symbol='triangle-down',size=11,color='#df5b59')))
    fig.update_layout(title='주가와 백테스트 매수·매도 시점',height=470,xaxis_title='날짜',yaxis_title='조정 가격',hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('매수·매도 표시는 실제 주문이 아니라 백테스트에서 가정한 다음 거래일 시가입니다.')
    c1,c2,c3=st.columns(3)
    c1.metric('최근 조정 종가',fmt(last.Close))
    pe=info.get('trailingPE');pb=info.get('priceToBook')
    c2.metric('현재 PER',f'{pe:.2f}배' if isinstance(pe,(int,float)) and np.isfinite(pe) else '자료 없음')
    c3.metric('현재 PBR',f'{pb:.2f}배' if isinstance(pb,(int,float)) and np.isfinite(pb) else '자료 없음')
    st.warning('10년 PER/PBR 밴드는 아직 구현되지 않았습니다. 실제 과거 EPS/BPS 시계열 확보 후 추가해야 하며, 현재 PER/PBR을 과거에 소급 적용하지 않습니다.')
with t2:
    days={'최근 1년':252,'최근 3년':756,'최대 10년':len(d)}[chart_days]
    v=d.tail(days)
    st.markdown('**읽는 법:** 초록색 조건은 매수 후보 필터를 충족했다는 뜻이며, 향후 상승을 보장하지 않습니다.')
    f=go.Figure()
    f.add_trace(go.Scatter(x=v.index,y=v.K,name='%K'))
    f.add_trace(go.Scatter(x=v.index,y=v.D,name='%D'))
    f.add_hline(y=oversold,line_dash='dot',annotation_text=f'과매도 {oversold}')
    f.add_hline(y=80,line_dash='dot',annotation_text='과매수 80')
    f.update_layout(title='스토캐스틱 (14,3,3) — 과매도권 상향교차 관찰',height=310)
    st.plotly_chart(f,use_container_width=True)
    f=go.Figure()
    f.add_trace(go.Scatter(x=v.index,y=v.MACD,name='MACD'))
    f.add_trace(go.Scatter(x=v.index,y=v.MACD_signal,name='Signal'))
    f.add_hline(y=0,line_dash='dot')
    f.update_layout(title='MACD (12,26,9) — MACD선이 Signal선을 상향 돌파하는지 확인',height=310)
    st.plotly_chart(f,use_container_width=True)
    f=go.Figure()
    for col,name in [('ADX','ADX'),('plusDI','+DI'),('minusDI','-DI')]:
        f.add_trace(go.Scatter(x=v.index,y=v[col],name=name))
    f.add_hline(y=adx_level,line_dash='dot',annotation_text=f'ADX {adx_level}')
    f.update_layout(title='ADX / +DI / -DI (14) — 추세 강도와 방향 확인',height=310)
    st.plotly_chart(f,use_container_width=True)
    st.info('ADX가 높아도 -DI가 +DI보다 높으면 하락 방향이 우세할 수 있습니다. ADX는 방향이 아니라 추세 강도를 나타냅니다.')
with t3:
    n=len(trades)
    bench=(d.Close/d.Close.iloc[0]).rename('매수 후 보유')
    # Strategy and benchmark are compared over identical observation dates.
    cols=st.columns(4)
    cols[0].metric('완료된 거래',f'{n}회')
    cols[1].metric('승률',f"{(trades['수익률(%)']>0).mean()*100:.1f}%" if n else '거래 없음')
    cols[2].metric('평균 거래 수익률',f"{trades['수익률(%)'].mean():+.2f}%" if n else '거래 없음')
    cols[3].metric('전략 MDD',f'{mdd(equity):.2f}%')
    c1,c2,c3=st.columns(3)
    c1.metric('전략 누적 수익률',f'{(equity.iloc[-1]-1)*100:+.2f}%')
    c2.metric('동일 종목 매수 후 보유',f'{(bench.iloc[-1]-1)*100:+.2f}%')
    c3.metric('매수 후 보유 MDD',f'{mdd(bench):.2f}%')
    f=go.Figure()
    f.add_trace(go.Scatter(x=equity.index,y=equity,name='3중 신호 전략'))
    f.add_trace(go.Scatter(x=bench.index,y=bench,name='동일 종목 매수 후 보유'))
    f.update_layout(title='자산곡선 비교 (초기 자산 = 1)',height=420,yaxis_title='자산 배수')
    st.plotly_chart(f,use_container_width=True)
    if n:
        st.subheader('장세별 완료 거래 성과')
        grouped=trades.groupby('장세',dropna=False)['수익률(%)'].agg(거래횟수='count',평균수익률='mean',승률=lambda x:(x>0).mean()*100).round(2)
        st.dataframe(grouped,use_container_width=True)
        st.caption('장세는 신호일의 종가와 200일 이동평균 비교로 분류합니다. 미래 장세를 미리 알았다고 가정하지 않습니다.')
        st.subheader('거래 내역')
        st.dataframe(trades.iloc[::-1],use_container_width=True,hide_index=True)
        st.download_button('거래 내역 CSV 다운로드',trades.to_csv(index=False).encode('utf-8-sig'),file_name=f'{ticker}_trades.csv',mime='text/csv')
    else:
        st.info('현재 조건에서는 완료된 거래가 없습니다. 신호 조건을 완화해 비교해 보세요.')
    st.caption('신호는 종가 확정 후 확인하며, 다음 거래일 시가에 진입하고 20거래일 뒤 시가에 청산합니다. 거래비용은 매수·매도 각각 반영합니다. 마지막 미완료 거래는 제외합니다.')
    st.warning('연구용 백테스트입니다. Yahoo Finance의 자동 조정 OHLC를 사용하므로 실제 체결가와 차이가 있습니다. 세금·슬리피지·유동성·배당 현금흐름·데이터 오류를 완전히 반영하지 않습니다. 수익률은 미래 성과를 보장하지 않습니다.')

with t4:
    render_scanner(indicators,get_signals,run_backtest,validity,hold,fee,oversold,adx_level)
