import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

US = '''AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO BRK-B JPM V MA UNH XOM COST WMT PG JNJ HD ABBV BAC KO PEP CRM ORCL AMD NFLX ADBE CSCO IBM INTC QCOM TXN AMAT MU LLY MRK CAT GE BA DIS NKE MCD SBUX GS MS C CVX COP UBER PLTR'''.split()
KR = '''005930.KS 000660.KS 005380.KS 000270.KS 035420.KS 035720.KS 051910.KS 006400.KS 207940.KS 068270.KS 005490.KS 012330.KS 028260.KS 105560.KS 055550.KS 086790.KS 032830.KS 000810.KS 003550.KS 066570.KS 034730.KS 096770.KS 017670.KS 030200.KS 015760.KS 009150.KS 018260.KS 010130.KS 011200.KS 003670.KS 247540.KQ 086520.KQ 196170.KQ 091990.KQ 263750.KQ 293490.KQ 041510.KQ 035900.KQ 122870.KQ 095340.KQ 058470.KQ 039030.KQ 214150.KQ 067310.KQ 036570.KS 259960.KS 352820.KS 373220.KS 402340.KS 012450.KS'''.split()

# Display names are labels, not trading signals. Unknown tickers retain their symbols.
US_NAMES = dict(zip(US, ["Apple","Microsoft","NVIDIA","Amazon","Alphabet","Meta Platforms","Tesla","Broadcom","Berkshire Hathaway","JPMorgan Chase","Visa","Mastercard","UnitedHealth Group","Exxon Mobil","Costco","Walmart","Procter & Gamble","Johnson & Johnson","Home Depot","AbbVie","Bank of America","Coca-Cola","PepsiCo","Salesforce","Oracle","Advanced Micro Devices","Netflix","Adobe","Cisco Systems","IBM","Intel","Qualcomm","Texas Instruments","Applied Materials","Micron Technology","Eli Lilly","Merck & Co.","Caterpillar","GE Aerospace","Boeing","Walt Disney","Nike","McDonald's","Starbucks","Goldman Sachs","Morgan Stanley","Citigroup","Chevron","ConocoPhillips","Uber Technologies","Palantir Technologies"]))
KR_NAMES_KO = dict(zip(KR, ["삼성전자","SK하이닉스","현대자동차","기아","NAVER","카카오","LG화학","삼성SDI","삼성바이오로직스","셀트리온","POSCO홀딩스","현대모비스","삼성물산","KB금융","신한지주","하나금융지주","삼성생명","삼성화재","LG","LG전자","SK","SK이노베이션","SK텔레콤","KT","한국전력","삼성전기","삼성에스디에스","고려아연","HMM","포스코퓨처엠","에코프로비엠","에코프로","알테오젠","셀트리온헬스케어","펄어비스","카카오게임즈","에스엠","JYP Ent.","와이지엔터테인먼트","ISC","리노공업","이오테크닉스","클래시스","에코프로에이치엔","엔씨소프트","크래프톤","하이브","LG에너지솔루션","두산에너빌리티","한화에어로스페이스"]))
KR_NAMES_EN = dict(zip(KR, ["Samsung Electronics","SK hynix","Hyundai Motor","Kia","NAVER","Kakao","LG Chem","Samsung SDI","Samsung Biologics","Celltrion","POSCO Holdings","Hyundai Mobis","Samsung C&T","KB Financial Group","Shinhan Financial Group","Hana Financial Group","Samsung Life Insurance","Samsung Fire & Marine Insurance","LG Corp.","LG Electronics","SK Inc.","SK Innovation","SK Telecom","KT Corp.","Korea Electric Power","Samsung Electro-Mechanics","Samsung SDS","Korea Zinc","HMM","POSCO Future M","Ecopro BM","Ecopro","Alteogen","Celltrion Healthcare","Pearl Abyss","Kakao Games","SM Entertainment","JYP Entertainment","YG Entertainment","ISC","LEENO Industrial","EO Technics","Classys","Ecopro HN","NCSoft","Krafton","HYBE","LG Energy Solution","Doosan Enerbility","Hanwha Aerospace"]))

def display_names(ticker):
    if ticker in US_NAMES:
        return US_NAMES[ticker], ''
    if ticker in KR_NAMES_EN:
        return KR_NAMES_EN[ticker], KR_NAMES_KO[ticker]
    return ticker, ''

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_batch(tickers, period='10y'):
    # Cached as a whole batch to avoid one HTTP call per ticker on every rerun.
    data = yf.download(list(tickers), period=period, auto_adjust=True, group_by='ticker', threads=False, progress=False, timeout=25)
    return data

def extract_frame(batch, ticker):
    if isinstance(batch.columns, pd.MultiIndex):
        if ticker in batch.columns.get_level_values(0):
            d = batch[ticker].copy()
        elif ticker in batch.columns.get_level_values(1):
            d = batch.xs(ticker, level=1, axis=1).copy()
        else:
            raise ValueError('티커 데이터 없음')
    else:
        d = batch.copy()
    needed=['Open','High','Low','Close','Volume']
    d=d[needed].dropna(subset=['Open','High','Low','Close'])
    if len(d)<220: raise ValueError('데이터 220일 미만')
    d.index=pd.to_datetime(d.index).tz_localize(None)
    return d

def scan_market(tickers, indicators, get_signals, run_backtest, validity, hold, fee, oversold, adx_level, progress=None):
    rows=[]; failures=[]
    groups=[tickers[i:i+10] for i in range(0,len(tickers),10)]
    for gi,group in enumerate(groups):
        try:
            batch=fetch_batch(tuple(group))
        except Exception as exc:
            failures.extend([{'티커':t,'오류':str(exc)[:110]} for t in group]); continue
        for ticker in group:
            try:
                df=extract_frame(batch,ticker)
                d=indicators(df,oversold,adx_level)
                s,m,a,combined=get_signals(d,validity)
                t,eq=run_backtest(d,combined,hold,fee)
                last=d.iloc[-1]
                benchmark=(d.Close/d.Close.iloc[0])
                strategy_return=(float(eq.iloc[-1])-1)*100
                strategy_mdd=float((eq/eq.cummax()-1).min()*100)
                benchmark_return=(float(benchmark.iloc[-1])-1)*100
                benchmark_mdd=float((benchmark/benchmark.cummax()-1).min()*100)
                yearly=t.copy()
                if len(yearly):
                    yearly['연도']=pd.to_datetime(yearly['매도일']).dt.year
                    yearly_counts=yearly.groupby('연도').size().to_dict()
                else:
                    yearly_counts={}
                recent=combined.tail(5)
                dates=d.index[combined.fillna(False)]
                latest=str(dates[-1].date()) if len(dates) else '없음'
                age=(len(d)-1-d.index.get_loc(dates[-1])) if len(dates) else None
                # Scoring is a transparent heuristic, NOT a predicted return or probability.
                flags=[bool(s.iloc[-1]),bool(m.iloc[-1]),bool(a.iloc[-1])]
                score=30*sum(flags)+5*bool(last.plusDI>last.minusDI)+5*bool(last.Close>last.SMA200 if pd.notna(last.SMA200) else False)
                english_name, korean_name = display_names(ticker)
                rows.append({'시장':'미국' if ticker in US else '한국','종목명(영문)':english_name,'종목명(한국어)':korean_name,'티커':ticker,'최종 데이터일':str(d.index[-1].date()),
                             '3중 신호': '현재 충족' if bool(combined.iloc[-1]) else ('최근 5일 발생' if bool(recent.any()) else '없음'),
                             '점수':score,'조건 충족':sum(flags),'스토캐스틱': '✓' if flags[0] else '—',
                             'MACD':'✓' if flags[1] else '—','ADX':'✓' if flags[2] else '—',
                             '과거 신호일 수':int(combined.sum()),'완료 거래':len(t),'전략 누적 수익률(%)':round(strategy_return,2),
                             '전략 MDD(%)':round(strategy_mdd,2),'매수보유 수익률(%)':round(benchmark_return,2),
                             '매수보유 MDD(%)':round(benchmark_mdd,2),
                             '상승장 거래':int((t['장세']=='상승 추세').sum()) if len(t) else 0,
                             '하락장 거래':int((t['장세']=='하락 추세').sum()) if len(t) else 0,
                             '연도별 거래수':str(yearly_counts),
                             '승률(%)':round(float((t['수익률(%)']>0).mean()*100),1) if len(t) else np.nan,
                             '평균 거래 수익률(%)':round(float(t['수익률(%)'].mean()),2) if len(t) else np.nan,
                             '마지막 신호일':latest,'신호 경과 거래일':age if age is not None else np.nan})
            except Exception as exc:
                failures.append({'티커':ticker,'오류':str(exc)[:110]})
        if progress: progress.progress((gi+1)/len(groups),text=f'종목 분석 {min((gi+1)*10,len(tickers))}/{len(tickers)}')
    return pd.DataFrame(rows),pd.DataFrame(failures)

def wilson_lower(p,n,z=1.96):
    if n<=0 or pd.isna(p): return np.nan
    center=p+z*z/(2*n)
    spread=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)
    return max(0,(center-spread)/(1+z*z/n))

def render_scanner(indicators,get_signals,run_backtest,validity,hold,fee,oversold,adx_level):
    st.subheader('🇺🇸 🇰🇷 한미 주식 3중 신호 스캐너')
    st.caption('보수형 기존 규칙을 그대로 적용합니다. 신호 랭킹과 조건 근접 랭킹을 분리하며, 점수는 예측 승률이 아닙니다.')
    universe=st.radio('검색 범위',['미국 50종목','한국 50종목','한미 100종목'],horizontal=True)
    pool=US if universe.startswith('미국') else KR if universe.startswith('한국') else US+KR
    extra=st.text_input('추가 티커 (선택, 쉼표로 구분)','',help='예: 035420.KS, AAPL')
    extras=[x.strip().upper() for x in extra.split(',') if x.strip()]
    tickers=list(dict.fromkeys(pool+extras))
    st.caption(f'선택된 {len(tickers)}종목 · 최초 스캔은 데이터 다운로드로 시간이 걸릴 수 있습니다. 동일 배치는 6시간 캐시합니다.')
    if st.button('🔎 전체 종목 스캔 시작',type='primary'):
        bar=st.progress(0)
        with st.spinner('과거 데이터 다운로드 및 신호 검사 중...'):
            result,failed=scan_market(tickers,indicators,get_signals,run_backtest,validity,hold,fee,oversold,adx_level,bar)
        bar.empty()
        st.session_state['scanner_results']=(result,failed,tuple(tickers),validity,hold,fee,oversold,adx_level)
    saved=st.session_state.get('scanner_results')
    if saved is None:
        st.info('스캔 시작 버튼을 눌러 실제 종목 데이터를 분석하세요. 임의의 순위를 표시하지 않습니다.')
        return
    result,failed,old_tickers,*old_params=saved
    if tuple(tickers)!=old_tickers or list(old_params)!=[validity,hold,fee,oversold,adx_level]:
        st.warning('종목 목록이나 전략 설정이 변경되었습니다. 새 설정을 적용하려면 다시 스캔하세요. 아래는 이전 결과입니다.')
    if result.empty:
        st.error('분석에 성공한 종목이 없습니다. Yahoo Finance 요청 제한 또는 네트워크 오류일 수 있습니다.')
    else:
        fresh=result[result['3중 신호']=='현재 충족'].sort_values(['점수','티커'],ascending=[False,True])
        historical=result[result['과거 신호일 수']>0]
        cols=st.columns(4)
        cols[0].metric('성공 분석',len(result))
        cols[1].metric('현재 3중 신호',len(fresh))
        cols[2].metric('과거 신호 발생 종목',len(historical))
        cols[3].metric('데이터 실패',len(failed))
        st.markdown('#### ① 현재 3중 신호 충족 종목')
        if fresh.empty: st.info('현재 충족 종목이 없습니다. 0개도 유효한 스캔 결과입니다.')
        else: st.dataframe(fresh,use_container_width=True,hide_index=True)
        st.markdown('#### ② 신호 근접 종목 랭킹')
        near=result[result['3중 신호']!='현재 충족'].sort_values(['점수','과거 신호일 수','티커'],ascending=[False,False,True])
        st.dataframe(near,use_container_width=True,hide_index=True)
        st.caption('점수 = 현재 스토캐스틱/MACD/ADX 조건별 30점 + +DI 우세 5점 + 200일선 위 5점. 최대 100점. 매수 추천이나 성공 확률이 아닙니다.')
        st.markdown('#### ③ 과거 신호 발생 및 백테스트 종목')
        st.dataframe(result.sort_values(['과거 신호일 수','완료 거래'],ascending=False),use_container_width=True,hide_index=True)
        st.markdown('#### ④ 보수형 전략 백테스트 신뢰성 랭킹')
        min_trades=st.select_slider('최소 완료 거래 횟수',options=[1,3,5,10,20],value=5,key='scanner_min_trades')
        eligible=result[result['완료 거래']>=min_trades].copy()
        if eligible.empty:
            st.info('선택한 최소 거래 횟수를 충족하는 종목이 없습니다. 표본 기준을 낮춰 확인할 수 있습니다.')
        else:
            eligible['승률 신뢰하한(%)']=eligible.apply(lambda r: wilson_lower(r['승률(%)']/100,int(r['완료 거래']))*100,axis=1).round(1)
            eligible['수익/낙폭 비율']=eligible['전략 누적 수익률(%)'] / eligible['전략 MDD(%)'].abs().replace(0,np.nan)
            st.caption('승률 신뢰하한은 독립적인 거래를 가정한 Wilson 95% 하한의 참고값입니다. 거래의 상관관계 때문에 실제 신뢰도를 과대평가할 수 있습니다.')
            show=['시장','종목명(영문)','종목명(한국어)','티커','완료 거래','승률(%)','승률 신뢰하한(%)','평균 거래 수익률(%)','전략 누적 수익률(%)','전략 MDD(%)','매수보유 수익률(%)','매수보유 MDD(%)']
            left,right=st.columns(2)
            with left:
                st.markdown('**승률 신뢰하한 순위**')
                st.dataframe(eligible.sort_values(['승률 신뢰하한(%)','완료 거래'],ascending=False)[show],hide_index=True,use_container_width=True)
            with right:
                st.markdown('**전략 누적 수익률 순위 (MDD 병기)**')
                st.dataframe(eligible.sort_values('전략 누적 수익률(%)',ascending=False)[show],hide_index=True,use_container_width=True)
            st.markdown('**미국·한국 시장별 비교 (종목별 평균, 동일가중)**')
            comparison=eligible.groupby('시장').agg(종목수=('티커','count'),총완료거래=('완료 거래','sum'),평균승률=('승률(%)','mean'),평균거래수익률=('평균 거래 수익률(%)','mean'),평균전략누적수익률=('전략 누적 수익률(%)','mean'),평균전략MDD=('전략 MDD(%)','mean')).round(2)
            st.dataframe(comparison,use_container_width=True)
            st.caption('시장별 수치는 종목별 결과의 단순 평균이며 실제 통합 포트폴리오 성과가 아닙니다. 상장폐지 종목 누락에 따른 생존 편향이 존재합니다.')
        st.markdown('#### ⑤ 개별 종목 거래 내역 및 연도별 결과')
        choice=st.selectbox('상세 분석 종목 선택',result['티커'].tolist(),format_func=lambda x: f"{display_names(x)[0]} ({x})",key='scanner_detail')
        if st.button('선택 종목 거래 내역 계산',key='scanner_detail_button'):
            try:
                detail_batch=fetch_batch((choice,))
                detail_d=indicators(extract_frame(detail_batch,choice),oversold,adx_level)
                _,_,_,detail_combined=get_signals(detail_d,validity)
                detail_trades,detail_eq=run_backtest(detail_d,detail_combined,hold,fee)
                if detail_trades.empty:
                    st.info('이 종목은 완료된 거래가 없습니다.')
                else:
                    detail_trades['매도연도']=pd.to_datetime(detail_trades['매도일']).dt.year
                    annual=detail_trades.groupby('매도연도')['수익률(%)'].agg(거래수='count',승률=lambda x:round((x>0).mean()*100,1),평균거래수익률='mean').round(2)
                    st.dataframe(annual,use_container_width=True)
                    st.dataframe(detail_trades.iloc[::-1],hide_index=True,use_container_width=True)
                    st.download_button('선택 종목 거래 CSV',detail_trades.to_csv(index=False).encode('utf-8-sig'),file_name=f'{choice}_trades.csv',mime='text/csv')
            except Exception as exc:
                st.error(f'상세 분석 실패: {exc}')
        st.warning('검증 주의: 거래별 다음날 시가 체결 및 편도 비용 가정을 사용합니다. 미완료 거래, 세금, 실제 슬리피지, 유동성은 완전히 반영되지 않습니다. 같은 종목의 반복 거래는 독립 표본이 아니며, 과거 성과는 미래 수익을 예측하지 않습니다.')
        st.download_button('전체 결과 CSV 저장',result.to_csv(index=False).encode('utf-8-sig'),'mystockquant_scan.csv','text/csv')
    if not failed.empty:
        with st.expander(f'데이터 실패 종목 {len(failed)}개 보기'):
            st.dataframe(failed,hide_index=True,use_container_width=True)
    st.warning('현재 스캔은 고정 종목군의 후향적 검사입니다. 상장폐지 종목 누락·종목 선정 편향·조정주가·거래비용·미래 정보 사용 여부를 추가 검증해야 전략 성과를 신뢰할 수 있습니다. 과거 신호일 수는 연속된 신호일을 각각 셀 수 있어 완료 거래 횟수와 다릅니다.')
