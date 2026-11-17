# Stock Analyzer Pro (MVP)

Python 3.10.11 supported.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publish with Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt` at its root.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch, and `app.py` as the entry point; deploy.
4. Open the generated HTTPS URL on desktop or mobile.

## Notes
- The application downloads Yahoo Finance data at runtime; network access is required.
- Historical PER/PBR bands are **not** yet implemented because point-in-time historical earnings and book values require a reliable financial-data feed.
- Price charts use adjusted OHLC data. Backtest buys at next adjusted open and exits after N sessions at adjusted open, which is an approximation.
- No trades can be taken on the last `hold+1` sessions because the future exit price is unavailable.
