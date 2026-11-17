# MyStockQuant 2.0

Streamlit으로 만든 주식 기술적 분석 및 최대 10년 백테스트 앱입니다.

## 기능
- 한국/미국 주식 티커 검색 (예: `005930.KS`, `MSFT`)
- 스토캐스틱(14,3,3) 과매도 상향교차, MACD(12,26,9) 상향교차, ADX(14) 추세 확인
- 최근 N거래일 내 이벤트 결합, 다음 거래일 시가 진입, 정해진 보유기간 이후 시가 청산
- 신호별 상태·최근 발생일·가격 차트의 거래 마커
- 승률, 거래 수, 평균 수익률, MDD, 누적 성과, 동일 종목 매수 후 보유 비교
- 신호 당시 200일 이동평균 기반 장세별 성과

## Streamlit Community Cloud 배포
GitHub 저장소 루트에 `app.py`, `requirements.txt`, `README.md`를 업로드합니다. 이미 배포된 앱은 GitHub의 `main` 브랜치에서 변경 사항을 반영합니다.

## 로컬 실행
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 주의사항
- Yahoo Finance의 요청 제한으로 데이터 로딩이 실패할 수 있습니다.
- PER/PBR의 과거 밴드는 구현되지 않았습니다. 과거 재무 데이터를 별도로 확보해야 합니다.
- 연구/교육용이며 투자 조언이 아닙니다.
