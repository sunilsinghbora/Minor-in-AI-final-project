from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)

# Mock ticker symbols for autocomplete
TICKER_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'ADBE',
    'ORCL', 'CRM', 'INTC', 'CSCO', 'IBM', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SPOT',
    'BABA', 'JD', 'NTES', 'BILI', 'PDD', 'TME', 'DIDI', 'NIO', 'XPEV', 'LI',
    'BA', 'GE', 'F', 'GM', 'CAT', 'DE', 'MMM', 'HON', 'UTX', 'LMT',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL'
]

def generate_mock_stock_data(ticker, days):
    """Generate mock stock data for a given ticker and number of days"""
    start_price = random.uniform(50, 300)
    current_price = start_price
    
    data = []
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i-1)
        # Add some realistic stock movement
        daily_change = random.uniform(-0.05, 0.05)  # -5% to +5% daily change
        current_price *= (1 + daily_change)
        current_price = max(current_price, 1)  # Minimum price of $1
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(current_price, 2),
            'volume': random.randint(1000000, 50000000)
        })
    
    return data

def generate_mock_prediction(last_price, days):
    """Generate mock prediction data"""
    current_price = last_price
    predictions = []
    
    for i in range(days):
        date = datetime.now() + timedelta(days=i+1)
        # Slight upward trend for predictions
        daily_change = random.uniform(-0.03, 0.05)
        current_price *= (1 + daily_change)
        
        predictions.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(current_price, 2)
        })
    
    return predictions

def calculate_sentiment_score(ticker):
    """Calculate mock sentiment score based on ticker"""
    # Mock sentiment based on ticker name hash for consistency
    hash_value = sum(ord(c) for c in ticker)
    sentiment = 30 + (hash_value % 40)  # Score between 30-70
    return sentiment

@app.route('/api/tickers/search', methods=['GET'])
def search_tickers():
    """Search for ticker symbols based on query"""
    query = request.args.get('q', '').upper()
    
    if not query:
        return jsonify([])
    
    matching_tickers = [ticker for ticker in TICKER_SYMBOLS if ticker.startswith(query)]
    return jsonify(matching_tickers[:10])  # Return max 10 results

@app.route('/api/stock/<ticker>/data', methods=['GET'])
def get_stock_data(ticker):
    """Get stock data for a specific ticker and time range"""
    ticker = ticker.upper()
    time_range = request.args.get('range', '1m')
    
    # Map time ranges to days
    days_map = {
        '5d': 5, '1w': 7, '1m': 30, '3m': 90, '6m': 180,
        '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, 'max': 2555
    }
    
    days = days_map.get(time_range, 30)
    data = generate_mock_stock_data(ticker, days)
    
    return jsonify({
        'ticker': ticker,
        'timeRange': time_range,
        'data': data
    })

@app.route('/api/stock/<ticker>/predict', methods=['POST'])
def predict_stock(ticker):
    """Predict stock prices using specified model"""
    ticker = ticker.upper()
    data = request.get_json()
    model = data.get('model', 'lstm')
    
    # Mock training progress updates would be handled by WebSocket in real app
    # For now, we'll just return prediction results
    
    # Get current stock data to use as baseline
    historical_data = generate_mock_stock_data(ticker, 90)
    last_price = historical_data[-1]['price']
    
    # Generate 3 months of predictions
    predictions = generate_mock_prediction(last_price, 90)
    
    # Mock model metrics
    metrics = {
        'rmse': round(random.uniform(1.5, 3.5), 2),
        'mae': round(random.uniform(1.0, 2.5), 2),
        'r2': round(random.uniform(0.85, 0.95), 3),
        'training_accuracy': round(random.uniform(90, 98), 1),
        'validation_accuracy': round(random.uniform(88, 96), 1),
        'epochs': random.randint(30, 80)
    }
    
    return jsonify({
        'ticker': ticker,
        'model': model,
        'predictions': predictions,
        'metrics': metrics
    })

@app.route('/api/stock/<ticker>/sentiment', methods=['GET'])
def get_sentiment(ticker):
    """Get sentiment analysis for a stock"""
    ticker = ticker.upper()
    score = calculate_sentiment_score(ticker)
    
    return jsonify({
        'ticker': ticker,
        'score': score,
        'description': 'Based on news analysis from the last 3 months',
        'sources_analyzed': random.randint(50, 200)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)