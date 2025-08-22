"""
Flask backend for the stock analysis app.

Responsibilities:
- Fetch historical prices and basic company info from Yahoo Finance.
- Train a small LSTM/GRU model on the fly and produce short-horizon forecasts.
- Report per-epoch training progress via an in-memory store.
- Provide endpoints for health, ticker search, data, predict, CSV, and sentiment.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import io
import logging
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Warm-up removed: no pre-initialization at startup to keep start time minimal

# Training progress store (keyed by client-provided progress_key)
# This is a simple in-memory map suitable for single-process dev usage.
# In production, use a shared store (Redis, DB) if you scale beyond one process.
TRAIN_PROGRESS = {}

# Mock ticker symbols for autocomplete
TICKER_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'ADBE',
    'ORCL', 'CRM', 'INTC', 'CSCO', 'IBM', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SPOT',
    'BABA', 'JD', 'NTES', 'BILI', 'PDD', 'TME', 'DIDI', 'NIO', 'XPEV', 'LI',
    'BA', 'GE', 'F', 'GM', 'CAT', 'DE', 'MMM', 'HON', 'UTX', 'LMT',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL'
]

# Mock data removed: backend now returns explicit errors when data fetch or training fails

def calculate_sentiment_score(ticker):
    """Deprecated: Calculate mock sentiment score based on ticker (kept for fallback)"""
    hash_value = sum(ord(c) for c in ticker)
    sentiment = 30 + (hash_value % 40)
    return sentiment

# ----------------------
# Real data integrations
# ----------------------

def _yahoo_time_params(time_range: str):
    """Map UI time range to Yahoo Finance chart API range and interval."""
    mapping = {
        '5d': ('5d', '1h'),
        '1w': ('7d', '1h'),
        '1m': ('1mo', '1d'),
        '3m': ('3mo', '1d'),
        '6m': ('6mo', '1d'),
        '1y': ('1y', '1d'),
        '2y': ('2y', '1d'),
        '3y': ('3y', '1d'),
        '5y': ('5y', '1wk'),
        '10y': ('10y', '1wk'),
        'max': ('max', '1mo'),
    }
    return mapping.get(time_range, ('1mo', '1d'))

def _time_range_to_unix_window(time_range: str):
    """Compute (period1, period2) unix timestamps for Yahoo CSV download based on range.

    For 'max', request full available history by setting period1=0.
    """
    now = datetime.utcnow()
    if time_range == 'max':
        return 0, int(now.timestamp())
    delta_map = {
        '5d': 5,
        '1w': 7,
        '1m': 30,
        '3m': 90,
        '6m': 180,
        '1y': 365,
        '2y': 730,
        '3y': 1095,
        '5y': 1825,
        '10y': 3650,
        '15y': 365*15,
        '20y': 365*20,
        '25y': 365*25,
        '30y': 365*30,
    }
    days = delta_map.get(time_range, 365)
    start = now - timedelta(days=days)
    period1 = int(start.timestamp())
    period2 = int(now.timestamp())
    return period1, period2

def fetch_company_info(ticker: str):
    """Fetch company name and quote basics from Yahoo Finance quote API."""
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={quote_plus(ticker)}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()
        result = (data or {}).get('quoteResponse', {}).get('result', [])
        if result:
            item = result[0]
            name = item.get('longName') or item.get('shortName') or ticker
            return {
                'symbol': item.get('symbol', ticker),
                'name': name
            }
    except Exception as e:
        app.logger.warning(f"Failed to fetch company info for {ticker}: {e}")
    return {'symbol': ticker, 'name': ticker}

def fetch_yahoo_chart_data(ticker: str, time_range: str):
    """Fetch historical price data from Yahoo Finance chart API."""
    rng, interval = _yahoo_time_params(time_range)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(ticker)}"
        f"?range={rng}&interval={interval}&includePrePost=false&events=div,splits"
    )
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    chart = resp.json()
    result = (chart or {}).get('chart', {}).get('result')
    if not result:
        raise ValueError('No chart data')
    r0 = result[0]
    timestamps = r0.get('timestamp') or []
    indicators = r0.get('indicators', {})
    # Prefer adjusted close (dividends/splits adjusted); fallback to raw close if needed
    adjclose = indicators.get('adjclose', [{}])[0].get('adjclose')
    quote0 = indicators.get('quote', [{}])[0]
    close = quote0.get('close')
    openp = quote0.get('open')
    prices = adjclose or close or []
    # Build data array
    data = []
    for i, (ts, px) in enumerate(zip(timestamps, prices)):
        if px is None:
            continue
        date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
        opv = None
        if openp and i < len(openp):
            opv = openp[i]
        item = {'date': date, 'price': round(float(px), 2)}
        if opv is not None:
            try:
                item['open'] = round(float(opv), 2)
            except Exception:
                pass
        data.append(item)
    return data

def fetch_yahoo_history_csv(ticker: str, time_range: str = '1y'):
    """Download historical data as CSV from Yahoo and return list of (date, open, adj_close|close)."""
    p1, p2 = _time_range_to_unix_window(time_range)
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{quote_plus(ticker)}"
        f"?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # Parse CSV into a DataFrame
    buf = io.BytesIO(r.content)
    df = pd.read_csv(buf)
    if df.empty or 'Date' not in df.columns:
        raise ValueError('Empty CSV from Yahoo Finance')
    # Prefer Adj Close, fallback to Close
    price_col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
    if not price_col:
        raise ValueError('CSV missing price columns')
    out = []
    for _, row in df.iterrows():
        dt = str(row['Date'])
        px = row[price_col]
        op = row['Open'] if 'Open' in df.columns else None
        if pd.isna(px):
            continue
        out.append((dt, None if pd.isna(op) else float(op), float(px)))
    if not out:
        raise ValueError('No prices parsed from CSV')
    return out, r.content

def fetch_yahoo_close_series(ticker: str, range_code: str = '1y'):
    """Fetch daily open and close prices (adjclose preferred for close) for model training.
    range_code examples: '6mo', '1y', '2y', '5y'.
    """
    # Yahoo chart API supports up to '10y' and 'max'. For larger custom ranges, use 'max'.
    rng_in = (range_code or '1y')
    try:
        if isinstance(rng_in, str) and rng_in.endswith('y'):
            yrs = int(rng_in[:-1])
            rng = 'max' if yrs > 10 else rng_in
        else:
            rng = rng_in
    except Exception:
        rng = '1y'
    interval = '1d'
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(ticker)}"
        f"?range={rng}&interval={interval}&includePrePost=false&events=div,splits"
    )
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    chart = resp.json()
    result = (chart or {}).get('chart', {}).get('result')
    if not result:
        raise ValueError('No chart data')
    r0 = result[0]
    timestamps = r0.get('timestamp') or []
    indicators = r0.get('indicators', {})
    adjclose = indicators.get('adjclose', [{}])[0].get('adjclose')
    quote = indicators.get('quote', [{}])[0]
    close = quote.get('close')
    openp = quote.get('open')
    prices = adjclose or close or []
    # Build tuples of (date, open, price)
    out = []
    for i, (ts, px) in enumerate(zip(timestamps, prices)):
        if px is None:
            continue
        date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
        op = None
        if openp and i < len(openp):
            op = openp[i]
        out.append((date, None if op is None else float(op), float(px)))
    if not out:
        raise ValueError('Empty price series')
    return out

POSITIVE_WORDS = {
    'beat', 'beats', 'surge', 'soar', 'rally', 'gain', 'gains', 'up', 'bull', 'bullish', 'record', 'strong',
    'outperform', 'upgrade', 'buy', 'overweight', 'profit', 'growth', 'optimistic', 'tops', 'exceed'
}
NEGATIVE_WORDS = {
    'miss', 'falls', 'drop', 'plunge', 'plunges', 'down', 'bear', 'bearish', 'weak', 'underperform', 'downgrade',
    'sell', 'underweight', 'loss', 'decline', 'warning', 'cuts', 'lawsuit', 'probe', 'slump'
}

def _score_text(text: str) -> int:
    t = (text or '').lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    # Normalize to 0-100 with 50 neutral baseline
    score = 50 + (pos - neg) * 10
    return max(0, min(100, score))

def fetch_sentiment_from_rss(ticker: str):
    """Fetch recent headlines from Yahoo Finance and Google News RSS and compute a simple sentiment score."""
    feeds = []
    # Yahoo Finance RSS for ticker
    yf_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"
    # Google News RSS for ticker + stock
    google_url = f"https://news.google.com/rss/search?q={quote_plus(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en"
    for url in (yf_url, google_url):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError:
                continue
            # RSS 2.0 typically has channel/item
            channel = root.find('channel')
            items = channel.findall('item') if channel is not None else root.findall('.//item')
            for it in items[:30]:  # limit items
                title = (it.findtext('title') or '').strip()
                desc = (it.findtext('description') or '').strip()
                link = (it.findtext('link') or '').strip()
                pub_date = (it.findtext('pubDate') or '').strip()
                feeds.append({
                    'title': title,
                    'description': desc,
                    'link': link,
                    'pubDate': pub_date,
                    'source': 'Yahoo Finance' if 'yahoo' in url else 'Google News'
                })
        except Exception as e:
            app.logger.warning(f"RSS fetch failed for {url}: {e}")

    if not feeds:
        # fallback to mock
        return calculate_sentiment_score(ticker), []

    # Compute average sentiment from titles + descriptions
    scores = []
    for item in feeds:
        s = _score_text(f"{item['title']} {item['description']}")
        scores.append(s)
    avg = int(round(sum(scores) / len(scores))) if scores else 50
    return avg, feeds[:20]

@app.route('/api/tickers/search', methods=['GET'])
def search_tickers():
    """Search for ticker symbols based on query using Yahoo Finance autocomplete."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    # Try newer search endpoint first; fallback to older autoc API; finally local heuristics
    search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={quote_plus(query)}&lang=en-US&region=US"
    auto_url = f"https://autoc.finance.yahoo.com/autoc?query={quote_plus(query)}&region=1&lang=en"
    try:
        out = []
        try:
            r = requests.get(search_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            data = r.json() or {}
            quotes = data.get('quotes') or []
            for it in quotes:
                symbol = it.get('symbol')
                name = it.get('shortname') or it.get('longname') or symbol
                type_disp = it.get('quoteType')
                exch = it.get('exchDisp') or it.get('exchange')
                if symbol:
                    out.append({'symbol': symbol, 'name': name, 'exchange': exch, 'type': type_disp})
        except Exception as inner_e:
            app.logger.info(f"Primary search endpoint failed, trying autoc: {inner_e}")
            r = requests.get(auto_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            data = r.json()
            results = (data or {}).get('ResultSet', {}).get('Result', [])
            for it in results:
                symbol = it.get('symbol')
                name = it.get('name') or symbol
                exch = it.get('exchDisp')
                type_disp = it.get('typeDisp')
                if symbol and type_disp in (None, 'Equity', 'ETF', 'Fund', 'Index'):
                    out.append({'symbol': symbol, 'name': name, 'exchange': exch, 'type': type_disp})
        # If nothing found, fall back to local list heuristics (no external dependency)
        if not out:
            q = query.upper()
            starts = [t for t in TICKER_SYMBOLS if t.startswith(q)]
            contains = [t for t in TICKER_SYMBOLS if q in t and t not in starts]
            fallback = starts + contains
            if fallback:
                return jsonify([{'symbol': t, 'name': t} for t in fallback[:10]])
        return jsonify(out[:10])
    except Exception as e:
        app.logger.warning(f"Autocomplete failed, falling back to mock list: {e}")
        # Fallback to mock list
        q = query.upper()
        matching = [t for t in TICKER_SYMBOLS if t.startswith(q)]
        return jsonify([{'symbol': t, 'name': t} for t in matching[:10]])

@app.route('/api/stock/<ticker>/data', methods=['GET'])
def get_stock_data(ticker):
    """Get stock data for a specific ticker and time range from Yahoo Finance."""
    ticker = ticker.upper()
    time_range = request.args.get('range', '1m')
    try:
        company = fetch_company_info(ticker)
        data = fetch_yahoo_chart_data(ticker, time_range)
        if not data:
            raise ValueError('Empty data from Yahoo Finance')
        return jsonify({
            'ticker': ticker,
            'companyName': company.get('name', ticker),
            'timeRange': time_range,
            'data': data
        })
    except Exception as e:
        app.logger.warning(f"Yahoo data failed for {ticker} ({time_range}): {e}")
        return jsonify({'error': 'Data fetch failed', 'details': str(e)}), 502

@app.route('/api/stock/<ticker>/predict', methods=['POST'])
def predict_stock(ticker):
    """Predict stock prices using specified model with real Yahoo Finance data.

    Trains a small model per request on the fly and produces short-horizon forecasts.
    Falls back to mock predictions if ML stack fails.
    """
    from importlib import import_module

    # 1) Parse request and normalize inputs
    ticker = ticker.upper()
    data = request.get_json(silent=True) or {}
    model_name = (data.get('model', 'lstm') or 'lstm').lower()
    if model_name not in ('lstm', 'gru'):
        model_name = 'lstm'
    expert_mode = bool(data.get('expertMode', False))
    progress_key = data.get('progress_key')
    # If expert mode: honor client-provided controls; else: use max data by default
    train_range = data.get('train_range', 'max' if not expert_mode else '6mo')
    # Normalize range: any year value > 20 maps to 'max' (all available history)
    try:
        if isinstance(train_range, str) and train_range.endswith('y'):
            yrs = int(train_range[:-1])
            if yrs > 20:
                train_range = 'max'
    except Exception:
        pass
    # Honor client-provided epochs; if missing/invalid, default to 20
    try:
        epochs_in = data.get('epochs', None)
        epochs = int(epochs_in) if epochs_in is not None else 20
        if epochs <= 0:
            epochs = 20
    except Exception:
        epochs = 20
    # Batch size: default 32 in both modes unless overridden
    batch_size = int(data.get('batch_size', 32))
    # Dropout: default 0.5; Expert Mode can override via request body
    default_dropout = 0.5
    dropout = float(data.get('dropout', default_dropout))
    # Forecast horizon in trading days (default 5)
    horizon = int(data.get('horizon', 5))
    # Window length for sliding windows (default 60)
    try:
        window = int(data.get('window', 60))
        if window < 10:
            window = 10
    except Exception:
        window = 60

    try:
        # 2) Fetch training data via CSV first; fall back to chart API on failure (e.g., 401)
        try:
            series, _csv_bytes = fetch_yahoo_history_csv(ticker, train_range)
        except Exception as fetch_e:
            app.logger.info(f"CSV fetch failed for {ticker} ({train_range}): {fetch_e}; trying chart API fallback")
            try:
                series = fetch_yahoo_close_series(ticker, range_code=train_range)
            except Exception as chart_e:
                raise chart_e
        dates, opens, closes = zip(*series)

        # 3) Log resolved configuration for transparency/debugging
        app.logger.info(
            f"Predict {ticker}: model={model_name} expert={expert_mode} epochs_req={data.get('epochs')} epochs={epochs} "
            f"range={train_range} horizon={horizon} batch={batch_size} dropout={dropout} key={progress_key}"
        )

        # 4) Lazy import ML pipeline and set a fixed train/test split (chronological 95/5)
        pipeline = import_module('ml.pipeline')
        # Always use a 95/5 train/test split: oldest 95% train, newest 5% test
        test_split = 0.05
        # 5) Initialize progress if key provided and create per-epoch hook
        if progress_key:
            TRAIN_PROGRESS[progress_key] = {
                'status': 'running',
                'current': 0,
                'total': int(epochs),
                'percent': 0,
                'loss': None,
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }

            def _progress_hook(current_epoch: int, total_epochs: int, loss: float | None):
                TRAIN_PROGRESS[progress_key] = {
                    **TRAIN_PROGRESS.get(progress_key, {}),
                    'status': 'running',
                    'current': int(current_epoch),
                    'total': int(total_epochs),
                    'percent': max(0, min(100, int(round(100.0 * current_epoch / max(total_epochs, 1))))),
                    'loss': None if loss is None else float(loss),
                    'updated_at': datetime.now().isoformat(),
                }
        else:
            _progress_hook = None

        # 6) Train with open+close features (target is close).
        #    Note: the pipeline now uses StandardScaler for both LSTM and GRU to normalize levels.
        result = pipeline.train_and_forecast(
            {'open': list(opens), 'close': list(closes)},
            model_name=model_name,
            window=window,
            horizon=horizon,
            epochs=epochs,
            verbose=0,
            test_split=test_split,
            dropout=dropout,
            batch_size=batch_size,
            progress_callback=_progress_hook,
        )
        preds = result['predictions']
        fitted = result.get('fitted') or []

        # 7) Build next dates after last available (skip weekends, no duplicates)
        last_dt = datetime.strptime(dates[-1], '%Y-%m-%d')
        d = last_dt
        out = []
        for p in preds:
            d = d + timedelta(days=1)
            while d.weekday() >= 5:
                d = d + timedelta(days=1)
            out.append({'date': d.strftime('%Y-%m-%d'), 'price': round(float(p), 2)})

        metrics = result.get('metrics', {})
        if progress_key:
            TRAIN_PROGRESS[progress_key] = {
                **TRAIN_PROGRESS.get(progress_key, {}),
                'status': 'done',
                'current': int(metrics.get('epochs') or epochs),
                'total': int(epochs),
                'percent': 100,
                'updated_at': datetime.now().isoformat(),
            }
        # 8) Align fitted with dates: fitted corresponds to windows ending points (len = len(dates)-window)
        fitted_points = []
        test_boundary = None
        try:
            w = int(result.get('window', 45))
            base = list(dates)
            if isinstance(fitted, list) and len(base) > w and len(fitted) == (len(base) - w):
                fit_dates = base[w:]
                fitted_points = [{'date': fit_dates[i], 'price': round(float(v), 2)} for i, v in enumerate(fitted)]
                # Compute test split boundary index to help frontend color-code test region
                if isinstance(test_split, (int, float)) and 0.0 < float(test_split) < 0.9:
                    lenX = len(fitted_points)
                    split_idx = int(lenX * (1.0 - float(test_split)))
                    split_idx = max(split_idx, 8)
                    split_idx = min(max(split_idx, 0), lenX)  # clamp
                    test_boundary = {
                        'index': int(split_idx),
                        'date': fitted_points[split_idx]['date'] if split_idx < lenX else (fitted_points[-1]['date'] if lenX else None)
                    }
        except Exception:
            fitted_points = []
            test_boundary = None

        # 9) Return API payload with predictions, fitted series, metrics and echo of used config
        return jsonify({
            'ticker': ticker,
            'model': model_name,
            'predictions': out,
            'fitted': fitted_points,
            'test_boundary': test_boundary,
            'metrics': metrics,
            'used': {
                'dropout': dropout,
                'epochs': epochs,
                'train_range': train_range,
                'batch_size': batch_size,
                'window': window,
                'horizon': horizon,
                'test_split': float(test_split),
                'features': result.get('features', 'close'),
                'scaler': metrics.get('scaler')
            },
            'progress_key': progress_key,
        })
    except Exception as e:
        app.logger.warning(f"ML prediction failed for {ticker} ({model_name}): {e}")
        if progress_key:
            TRAIN_PROGRESS[progress_key] = {
                **TRAIN_PROGRESS.get(progress_key, {}),
                'status': 'error',
                'percent': TRAIN_PROGRESS.get(progress_key, {}).get('percent', 0),
                'updated_at': datetime.now().isoformat(),
                'error': str(e),
            }
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/api/progress/<key>', methods=['GET'])
def get_progress(key: str):
    """Return training progress for a given progress key."""
    info = TRAIN_PROGRESS.get(key)
    if not info:
        return jsonify({'status': 'unknown'}), 404
    return jsonify({
        'key': key,
        **info
    })

@app.route('/api/stock/<ticker>/csv', methods=['GET'])
def download_csv(ticker):
    """Download historical data as CSV for a ticker and time range."""
    ticker = ticker.upper()
    time_range = request.args.get('range', '1y')
    try:
        _series, csv_bytes = fetch_yahoo_history_csv(ticker, time_range)
        from flask import Response
        resp = Response(csv_bytes, mimetype='text/csv')
        resp.headers['Content-Disposition'] = f'attachment; filename="{ticker}_{time_range}.csv"'
        return resp
    except Exception as e:
        app.logger.warning(f"CSV download failed for {ticker} ({time_range}): {e}")
    # Fallback: build CSV from chart API data so callers still get data for ranges like 'max'
        try:
            data = fetch_yahoo_chart_data(ticker, time_range)
            # Synthesize CSV with Date,Close headers
            import csv
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["Date", "Close"])
            for row in data:
                writer.writerow([row.get('date'), row.get('price')])
            csv_text = buf.getvalue().encode('utf-8')
            from flask import Response
            resp = Response(csv_text, mimetype='text/csv')
            resp.headers['Content-Disposition'] = f'attachment; filename="{ticker}_{time_range}_chart.csv"'
            return resp
        except Exception as e2:
            app.logger.warning(f"CSV fallback via chart API also failed for {ticker} ({time_range}): {e2}")
            return jsonify({'error': 'CSV download failed', 'details': str(e)}), 502

@app.route('/api/stock/<ticker>/sentiment', methods=['GET'])
def get_sentiment(ticker):
    """Get sentiment analysis for a stock using Yahoo Finance and Google News RSS."""
    ticker = ticker.upper()
    try:
        score, articles = fetch_sentiment_from_rss(ticker)
        return jsonify({
            'ticker': ticker,
            'score': score,
            'description': 'Headline sentiment from Yahoo Finance and Google News (last 24-72h)',
            'sources_analyzed': len(articles),
            'articles': articles
        })
    except Exception as e:
        app.logger.warning(f"Sentiment fetch failed for {ticker}: {e}; using fallback")
        score = calculate_sentiment_score(ticker)
        return jsonify({
            'ticker': ticker,
            'score': score,
            'description': 'Fallback mock sentiment due to RSS error',
            'sources_analyzed': 0
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Warm-up health endpoint removed

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Run the development server (single process). In production, use a real WSGI server.
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)