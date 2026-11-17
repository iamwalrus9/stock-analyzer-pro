import numpy as np
import pandas as pd
import streamlit as st
from scanner import US, KR, display_names, fetch_batch, extract_frame

# Research hypotheses, not a verified reproduction of any educator's proprietary rules.
PERIODS=((5,3,3),(10,6,6),(20,12,12))

def stoch_three(df):
    d=df.copy()
    for label,(period,smooth_k,smooth_d) in zip(('short','mid','long'),PERIODS):
        low=d.Low.rolling(period,min_periods=period).min()
        high=d.High.rolling(period,min_periods=period).max()
        raw=100*(d.Close-low)/(high-low).replace(0,np.nan)
        k=raw.rolling(smooth_k,min_periods=smooth_k).mean()
        signal=k.rolling(smooth_d,min_periods=smooth_d).mean()
        d[label+'_k']=k
        d[label+'_d']=signal
        d[label+'_cross']=((k.shift(1)<=signal.shift(1))&(k>signal)).fillna(False)
    return d

def build_signals(d,window=10,oversold=30):
    # Variant A: long crossover, then mid, then short, all within a rolling window.
    # Event sequence must be strictly ordered (no same-day crossings).
    events={name:np.flatnonzero(d[name+'_cross'].to_numpy()) for name in ('short','mid','long')}
    sequential=np.zeros(len(d),dtype=bool)
    simultaneous=np.zeros(len(d),dtype=bool)
    for i in range(len(d)):
        recent_long=events['long'][(events['long']>=i-window+1)&(events['long']<=i)]
        recent_mid=events['mid'][(events['mid']>=i-window+1)&(events['mid']<=i)]
        recent_short=events['short'][(events['short']>=i-window+1)&(events['short']<=i)]
        # Require a short crossover on the signal date; avoid repeated daily triggers.
        if len(recent_short) and recent_short[-1]==i:
            sequential[i]=any(l<m<i for l in recent_long for m in recent_mid)
            simultaneous[i]=bool(len(recent_long) and len(recent_mid))
        # Oversold screen: short %K or %D was at/below threshold immediately before trigger.
        if not (pd.notna(d.short_k.iloc[i-1]) and pd.notna(d.short_d.iloc[i-1]) and
                min(d.short_k.iloc[i-1],d.short_d.iloc[i-1])<=oversold) if i>0 else True:
            sequential[i]=False
            simultaneous[i]=False
    return pd.Series(sequential,index=d.index),pd.Series(simultaneous,index=d.index)

def trade_test(d,signal,hold=20,fee_bps=10):
    # Non-overlapping trades; close-confirmed signals, next-session open fills.
    n=len(d); curve=np.ones(n); cash=1.0; trades=[]; i=0
    cost=fee_bps/10000
    while i<n:
        if bool(signal.iloc[i]) and i+hold+1<n:
            buy_i=i+1; sell_i=buy_i+hold
            buy=float(d.Open.iloc[buy_i]); sell=float(d.Open.iloc[sell_i])
            if np.isfinite(buy) and np.isfinite(sell) and buy>0 and sell>0:
                curve[i]=cash
                shares=cash*(1-cost)/buy
                for j in range(buy_i,sell_i):
                    curve[j]=shares*float(d.Close.iloc[j])
                cash=shares*sell*(1-cost)
                curve[sell_i]=cash
                trades.append({'신호일':str(d.index[i].date()),'매수일':str(d.index[buy_i].date()),
                               '매도일':str(d.index[sell_i].date()),'수익률(%)':((sell/buy)*(1-cost)**2-1)*100})
                i=sell_i+1
                continue
        curve[i]=cash
        i+=1
    return pd.DataFrame(trades),pd.Series(curve,index=d.index)

def metrics(t,eq,d,variant,ticker):
    n=len(t)
    benchmark=d.Close/d.Close.iloc[0]
    years=(d.index[-1]-d.index[0]).days/365.25
    return {'시장':'미국' if ticker in US else '한국','종목명(영문)':display_names(ticker)[0],
            '종목명(한국어)':display_names(ticker)[1],'티커':ticker,'전략':variant,
            '완료 거래':n,'승률(%)':round((t['수익률(%)']>0).mean()*100,2) if n else np.nan,
            '평균 거래 수익률(%)':round(t['수익률(%)'].mean(),2) if n else np.nan,
            '전략 누적 수익률(%)':round((eq.iloc[-1]-1)*100,2),
            '전략 CAGR(%)':round((eq.iloc[-1]**(1/years)-1)*100,2) if years>0 else np.nan,
            '전략 MDD(%)':round((eq/eq.cummax()-1).min()*100,2),
            '매수보유 CAGR(%)':round((benchmark.iloc[-1]**(1/years)-1)*100,2) if years>0 else np.nan}

def render_stoch_comparison():
    st.subheader('📊 슬로우 스토캐스틱 3개 전략 A/B 비교')
    st.caption('연구용 규칙: 단기 (5,3,3), 중기 (10,6,6), 장기 (20,12,12). 특정 전문가의 공식 매매법과 동일하다고 검증된 규칙은 아닙니다.')
    a,b,c=st.columns(3)
    with a: window=st.slider('신호 허용 기간 (거래일)',3,30,10,key='st_window')
    with b: oversold=st.slider('단기 과매도 기준',10,50,30,5,key='st_os')
    with c: hold=st.selectbox('보유기간 (거래일)',[5,10,20,40],index=2,key='st_hold')
    fee=st.number_input('편도 거래비용 (bps)',0,100,10,key='st_fee')
    universe=st.radio('시장',['미국 50종목','한국 50종목','한미 100종목'],horizontal=True,key='st_market')
    pool=US if universe.startswith('미국') else KR if universe.startswith('한국') else US+KR
    st.info('A: 장기→중기→단기 골든크로스 순차 발생. B: 세 골든크로스가 기간 내 모두 발생하고 단기 교차로 마무리. 두 전략 모두 단기 과매도 조건을 적용합니다.')
    if st.button('두 전략 10년 백테스트 실행',type='primary',key='st_run'):
        records=[];errors=[];bar=st.progress(0)
        for offset in range(0,len(pool),10):
            chunk=pool[offset:offset+10]
            try: batch=fetch_batch(tuple(chunk))
            except Exception as exc:
                errors.extend([(x,str(exc)) for x in chunk])
                bar.progress(min(1,(offset+len(chunk))/len(pool)))
                continue
            for ticker in chunk:
                try:
                    d=stoch_three(extract_frame(batch,ticker))
                    sa,sb=build_signals(d,window,oversold)
                    for name,sig in [('A 순차',sa),('B 3중 교차',sb)]:
                        t,eq=trade_test(d,sig,hold,fee)
                        row=metrics(t,eq,d,name,ticker)
                        row['과거 신호일 수']=int(sig.sum())
                        records.append(row)
                except Exception as exc: errors.append((ticker,str(exc)))
            bar.progress(min(1,(offset+len(chunk))/len(pool)))
        bar.empty()
        st.session_state['st_result']=(pd.DataFrame(records),errors,(universe,window,oversold,hold,fee))
    saved=st.session_state.get('st_result')
    if saved is None:
        st.caption('실제 시세를 다운로드하여 계산하므로 실행 전에는 승률을 표시하지 않습니다.')
        return
    result,errors,settings=saved
    if settings!=(universe,window,oversold,hold,fee):
        st.warning('설정이 변경됐습니다. 결과는 이전 설정 기준입니다. 다시 실행해 주세요.')
    if result.empty:
        st.error('분석 결과가 없습니다. 데이터 요청 제한을 확인하세요.')
        return
    st.metric('분석 성공 종목',result['티커'].nunique())
    st.metric('데이터 실패 종목',len(errors))
    summary=result.groupby('전략').agg(종목수=('티커','nunique'),완료거래=('완료 거래','sum'),
        거래발생종목=('완료 거래',lambda x:int((x>0).sum())),
        평균종목별승률=('승률(%)','mean'),평균거래수익률=('평균 거래 수익률(%)','mean'),
        평균전략CAGR=('전략 CAGR(%)','mean'),평균전략MDD=('전략 MDD(%)','mean')).round(2)
    st.markdown('#### A/B 전체 비교')
    st.dataframe(summary,use_container_width=True)
    st.caption('평균 승률은 거래가 있는 종목의 단순 평균입니다. 평균 CAGR은 개별 종목별 독립 운용 결과의 평균이며 통합 포트폴리오 수익률이 아닙니다.')
    minimum=st.select_slider('종목별 최소 거래 횟수',options=[1,3,5,10,20],value=5,key='st_min')
    filtered=result[result['완료 거래']>=minimum]
    st.markdown('#### 종목별 전략 랭킹')
    st.dataframe(filtered.sort_values(['전략 CAGR(%)','완료 거래'],ascending=[False,False]),
                 hide_index=True,use_container_width=True)
    st.markdown('#### 시장별 비교')
    st.dataframe(result.groupby(['시장','전략']).agg(완료거래=('완료 거래','sum'),
        평균거래수익률=('평균 거래 수익률(%)','mean'),평균CAGR=('전략 CAGR(%)','mean')).round(2),
        use_container_width=True)
    st.download_button('전체 결과 CSV',result.to_csv(index=False).encode('utf-8-sig'),
                       'stochastic_comparison.csv','text/csv')
    if errors:
        with st.expander('실패 종목 보기'):
            st.dataframe(pd.DataFrame(errors,columns=['티커','오류']),hide_index=True)
    st.warning('주의: 후향적 고정 종목군에는 생존 편향이 있습니다. 동일 데이터에서 규칙을 선택하면 과최적화 위험이 있으며, 거래 간 독립성도 보장되지 않습니다. 실제 투자 전 기간 외 검증과 포트폴리오 단위 검증이 필요합니다.')
