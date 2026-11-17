import numpy as np
import pandas as pd
import streamlit as st
from scanner import US, KR, fetch_batch, extract_frame, display_names
from stoch_compare import stoch_three, build_signals

def evaluate(d, signal, variant, hold, fee_bps, atr_mult=2.5):
    """Close-of-day signals; next open orders. Stop signals evaluated at close, executed next open."""
    close=d.Close.to_numpy(dtype=float); op=d.Open.to_numpy(dtype=float)
    n=len(d); curve=np.ones(n); cash=1.; entry=None; shares=0.; entry_price=0.; entry_i=0
    highest=0.; stop=0.; exit_next=False; trades=[]
    sma10=d.Close.rolling(10).mean().to_numpy()
    sma20=d.Close.rolling(20).mean().to_numpy()
    sma200=d.Close.rolling(200).mean().to_numpy()
    prev=d.Close.shift()
    tr=pd.concat([(d.High-d.Low),(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean().to_numpy()
    cost=fee_bps/10000
    for i in range(n):
        # Exit orders from previous close, at today's open.
        if entry is not None and exit_next:
            if np.isfinite(op[i]) and op[i]>0:
                cash=shares*op[i]*(1-cost)
                trades.append({'매수일':str(d.index[entry].date()),'매도일':str(d.index[i].date()),
                               '수익률(%)':((op[i]/entry_price)*(1-cost)**2-1)*100,
                               '검증구간':'검증 2023+' if d.index[entry]>=pd.Timestamp('2023-01-01') else '개발 ~2022'})
                entry=None;shares=0.;exit_next=False
        # Entries are only allowed on previous close signal, no same-day reentry.
        if entry is None and i>0 and bool(signal.iloc[i-1]):
            allowed=True
            if variant=='B + 200일선':
                allowed=np.isfinite(sma200[i-1]) and close[i-1]>sma200[i-1]
            if allowed and np.isfinite(op[i]) and op[i]>0:
                entry=i;entry_i=i;entry_price=op[i]
                shares=cash*(1-cost)/op[i]
                highest=close[i];stop=entry_price-atr_mult*atr[i-1] if np.isfinite(atr[i-1]) else -np.inf
        if entry is not None:
            highest=max(highest,close[i])
            age=i-entry_i
            if variant in ('B 20일 고정','B + 200일선'):
                exit_next=age>=hold-1
            elif variant=='B + MA10 매도':
                exit_next=(age>=1 and np.isfinite(sma10[i]) and close[i]<sma10[i]) or age>=59
            elif variant=='B + MA20 매도':
                exit_next=(age>=1 and np.isfinite(sma20[i]) and close[i]<sma20[i]) or age>=59
            elif variant=='B + ATR 손절':
                exit_next=(np.isfinite(atr[i]) and close[i]<stop) or age>=hold-1
            elif variant=='B + ATR 추적':
                if np.isfinite(atr[i]):
                    stop=max(stop,highest-atr_mult*atr[i])
                exit_next=close[i]<stop or age>=59
            curve[i]=shares*close[i]
        else:
            curve[i]=cash
    # Mark open position to market at last close for CAGR/MDD, exclude unfinished trades from win rate.
    years=(d.index[-1]-d.index[0]).days/365.25
    t=pd.DataFrame(trades)
    eq=pd.Series(curve,index=d.index)
    return {'완료거래':len(t),'승률(%)':round((t['수익률(%)']>0).mean()*100,2) if len(t) else np.nan,
            '평균거래수익률(%)':round(t['수익률(%)'].mean(),2) if len(t) else np.nan,
            'CAGR(%)':round((eq.iloc[-1]**(1/years)-1)*100,2) if years>0 and eq.iloc[-1]>0 else np.nan,
            'MDD(%)':round((eq/eq.cummax()-1).min()*100,2),
            '미청산포지션':int(entry is not None)},t

def render_exit_lab():
    st.subheader('🧪 B 전략 매도·시장 필터 실험')
    st.caption('매수 신호는 기존 B 3중 골든크로스를 유지합니다. 한 번에 매도 규칙 하나씩 변경해 비교합니다.')
    st.warning('2023년 이후 구간은 검증용 구간으로 표시하지만, 이미 이전 분석에서 같은 데이터를 살펴보았으므로 엄밀한 미사용(out-of-sample) 데이터는 아닙니다.')
    a,b,c=st.columns(3)
    with a: window=st.slider('골든크로스 허용기간',3,30,10,key='lab_win')
    with b: os=st.slider('단기 과매도',10,50,30,5,key='lab_os')
    with c: hold=st.selectbox('기준 보유기간',[10,20,40],index=1,key='lab_hold')
    fee=st.number_input('편도 거래비용 bps',0,100,10,key='lab_fee')
    atrmult=st.select_slider('ATR 손절 배수',[1.5,2.0,2.5,3.0,3.5],value=2.5,key='lab_atr')
    market=st.radio('대상',['미국 50종목','한국 50종목','한미 100종목'],horizontal=True,key='lab_market')
    pool=US if market.startswith('미국') else KR if market.startswith('한국') else US+KR
    variants=['B 20일 고정','B + 200일선','B + MA10 매도','B + MA20 매도','B + ATR 손절','B + ATR 추적']
    settings=(window,os,hold,fee,atrmult,market)
    if st.button('6가지 실험 백테스트 실행',type='primary',key='lab_run'):
        rows=[];fails=[];bar=st.progress(0)
        for offset in range(0,len(pool),10):
            chunk=pool[offset:offset+10]
            try:
                batch=fetch_batch(tuple(chunk))
                for ticker in chunk:
                    try:
                        d=stoch_three(extract_frame(batch,ticker))
                        _,signal=build_signals(d,window,os)
                        for v in variants:
                            m,t=evaluate(d,signal,v,hold,fee,atrmult)
                            rows.append({'시장':'미국' if ticker in US else '한국','종목':display_names(ticker)[0],
                                         '티커':ticker,'실험':v,**m})
                    except Exception as exc: fails.append((ticker,str(exc)))
            except Exception as exc: fails.extend((x,str(exc)) for x in chunk)
            bar.progress(min(1,(offset+len(chunk))/len(pool)))
        bar.empty()
        st.session_state['exit_lab']=(pd.DataFrame(rows),fails,settings)
    saved=st.session_state.get('exit_lab')
    if saved is None:
        st.info('실행 후 실제 과거 데이터 기반 결과가 표시됩니다.')
        return
    data,fails,old=saved
    if old!=settings: st.warning('설정이 변경됐습니다. 표시 결과는 이전 설정 기준입니다.')
    if data.empty:
        st.error('계산 결과가 없습니다. 데이터 실패 내역을 확인하세요.')
        return
    st.metric('분석 성공 종목',data['티커'].nunique())
    st.markdown('#### 전략별 전체 비교')
    agg=data.groupby('실험').agg(종목수=('티커','nunique'),완료거래=('완료거래','sum'),
        평균승률=('승률(%)','mean'),평균거래수익률=('평균거래수익률(%)','mean'),
        평균CAGR=('CAGR(%)','mean'),평균MDD=('MDD(%)','mean')).round(2)
    st.dataframe(agg,use_container_width=True)
    st.caption('평균 CAGR/MDD는 각 종목을 독립 운용한 값의 단순 평균입니다. 실제 통합 포트폴리오 성과가 아닙니다.')
    st.markdown('#### 미국·한국 비교')
    st.dataframe(data.groupby(['시장','실험']).agg(완료거래=('완료거래','sum'),
        평균CAGR=('CAGR(%)','mean'),평균MDD=('MDD(%)','mean')).round(2),use_container_width=True)
    min_n=st.select_slider('최소 완료 거래',options=[1,3,5,10,20],value=5,key='lab_min')
    st.markdown('#### 종목별 랭킹')
    st.dataframe(data[data['완료거래']>=min_n].sort_values('CAGR(%)',ascending=False),
                 hide_index=True,use_container_width=True)
    st.download_button('실험 결과 CSV',data.to_csv(index=False).encode('utf-8-sig'),
                       'exit_experiments.csv','text/csv')
    if fails:
        with st.expander(f'데이터 실패 {len(fails)}건'):
            st.dataframe(pd.DataFrame(fails,columns=['티커','오류']),hide_index=True)
    st.warning('한계: 현재 상장 종목만 포함해 생존 편향이 있습니다. 일봉 종가 기반 손절은 다음날 시가에 청산하므로 갭 하락 손실을 막지 못합니다. 거래 비용 외 세금·슬리피지는 미반영이며, 시장지수 200일선 필터가 아니라 개별 종목 200일선 필터입니다. 이 결과로 실거래를 결정하지 마세요.')
