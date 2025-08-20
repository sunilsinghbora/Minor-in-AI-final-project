import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, MultiHeadAttention, LayerNormalization
from sklearn.preprocessing import MinMaxScaler
import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import re

# Page configuration
st.set_page_config(
    page_title="Ticker Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for GitHub-style UI
def load_css():
    st.markdown("""
    <style>
    /* GitHub-style theme */
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Dark mode toggle */
    .dark-mode {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    .light-mode {
        background-color: #ffffff;
        color: #24292f;
    }
    
    /* Center the main content */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #d0d7de;
        padding: 8px 12px;
        font-size: 14px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2da44e;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2c974b;
    }
    
    /* News card styling */
    .news-card {
        border-radius: 6px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #d0d7de;
    }
    
    .news-card.positive {
        background-color: #dcfce7;
        border-color: #22c55e;
    }
    
    .news-card.negative {
        background-color: #fef2f2;
        border-color: #ef4444;
    }
    
    .news-card.neutral {
        background-color: #f8f9fa;
        border-color: #6b7280;
    }
    
    /* Chart container */
    .chart-container {
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 16px;
        margin: 16px 0;
    }
    
    /* Error message styling */
    .error-message {
        color: #d73a49;
        font-size: 14px;
        margin-top: 4px;
    }
    
    /* Loading spinner */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'ticker_suggestions' not in st.session_state:
        st.session_state.ticker_suggestions = []
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = ""

# Get popular tickers for suggestions
@st.cache_data(ttl=3600)
def get_popular_tickers():
    """Get list of popular stock tickers"""
    popular_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 
        'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'UBER', 'LYFT', 'SNAP',
        'SPOT', 'SQ', 'PYPL', 'SHOP', 'TWTR', 'ZOOM', 'DOCU', 'WORK',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'BTC-USD', 'ETH-USD'
    ]
    return popular_tickers

# Mock data for demonstration when Yahoo Finance is not accessible
def get_mock_stock_data(ticker, period="1y"):
    """Generate mock stock data for demonstration"""
    import datetime
    
    # Generate date range
    end_date = datetime.datetime.now()
    if period == "1wk":
        start_date = end_date - datetime.timedelta(weeks=1)
    elif period == "1mo":
        start_date = end_date - datetime.timedelta(days=30)
    elif period == "3mo":
        start_date = end_date - datetime.timedelta(days=90)
    elif period == "6mo":
        start_date = end_date - datetime.timedelta(days=180)
    elif period == "1y":
        start_date = end_date - datetime.timedelta(days=365)
    elif period == "3y":
        start_date = end_date - datetime.timedelta(days=1095)
    elif period == "5y":
        start_date = end_date - datetime.timedelta(days=1825)
    else:  # max
        start_date = end_date - datetime.timedelta(days=3650)
    
    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # Base price varies by ticker
    base_prices = {
        'AAPL': 150, 'TSLA': 200, 'GOOGL': 2500, 'MSFT': 300,
        'AMZN': 3000, 'META': 250, 'NVDA': 400, 'NFLX': 350
    }
    base_price = base_prices.get(ticker, 100)
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible demo data
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Create DataFrame similar to yfinance format
    data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'Close': prices,
        'Volume': [int(1000000 * (1 + np.random.normal(0, 0.5))) for _ in prices]
    }, index=dates)
    
    # Ensure High >= Close >= Low and High >= Open >= Low
    for i in range(len(data)):
        high = max(data.iloc[i]['Open'], data.iloc[i]['Close']) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(data.iloc[i]['Open'], data.iloc[i]['Close']) * (1 - abs(np.random.normal(0, 0.01)))
        data.iloc[i, data.columns.get_loc('High')] = high
        data.iloc[i, data.columns.get_loc('Low')] = low
    
    return data

# Validate ticker symbol
def validate_ticker(ticker):
    """Validate if ticker symbol exists"""
    # List of valid demo tickers
    valid_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 
        'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'UBER', 'LYFT', 'SNAP',
        'SPOT', 'SQ', 'PYPL', 'SHOP', 'ZOOM', 'DOCU',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'
    ]
    
    if ticker.upper() in valid_tickers:
        return True, None
    else:
        return False, f"Demo mode: Please use one of these tickers: {', '.join(valid_tickers[:10])}..."

# Fetch stock data
def fetch_stock_data_no_cache(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance or use mock data"""
    # Always use mock data for demonstration in this environment
    try:
        data = get_mock_stock_data(ticker, period)
        return data, "Using demo data (Network access limited in this environment)"
    except Exception as e:
        return None, f"Error generating demo data: {str(e)}"

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1y"):
    """Cached version of fetch_stock_data"""
    return fetch_stock_data_no_cache(ticker, period)

# Get ticker suggestions
def get_ticker_suggestions(query):
    """Get ticker suggestions based on user input"""
    popular = get_popular_tickers()
    if not query:
        return popular[:10]
    
    query = query.upper()
    suggestions = [t for t in popular if query in t]
    return suggestions[:10]

# Fetch stock data
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None, "No data found for this ticker"
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Prepare data for ML models
def prepare_ml_data(data, lookback=60):
    """Prepare data for machine learning models"""
    if len(data) < lookback + 10:
        return None, None, None, "Insufficient data for forecasting"
    
    # Use closing prices
    prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_prices)):
        X.append(scaled_prices[i-lookback:i, 0])
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return (X_train, X_test, y_train, y_test), scaler, None

# Create LSTM model
def create_lstm_model(input_shape):
    """Create LSTM model for price prediction"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create GRU model
def create_gru_model(input_shape):
    """Create GRU model for price prediction"""
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create Transformer-like model
def create_transformer_model(input_shape):
    """Create a simple transformer-like model"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Reshape for attention
    x = tf.keras.layers.Reshape((input_shape[0], 1))(inputs)
    
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x, x)
    
    # Add & Norm
    x = LayerNormalization()(x + attention_output)
    
    # Feed Forward
    ff_output = Dense(64, activation='relu')(x)
    ff_output = Dense(1)(ff_output)
    
    # Add & Norm
    x = LayerNormalization()(x + ff_output)
    
    # Global pooling and final dense
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(25, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Train model and make predictions
@st.cache_data(ttl=1800)
def train_and_predict(model_type, data, ticker, lookback=60):
    """Train model and make predictions"""
    ml_data, scaler, error = prepare_ml_data(data, lookback)
    if error:
        return None, None, error
    
    X_train, X_test, y_train, y_test = ml_data
    
    # Create model
    input_shape = (X_train.shape[1], 1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    if model_type == "LSTM":
        model = create_lstm_model(input_shape)
    elif model_type == "GRU":
        model = create_gru_model(input_shape)
    else:  # Transformer
        model = create_transformer_model(input_shape)
    
    # Train with few epochs for demo
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Make predictions for next 5 days
    last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]
    predictions = []
    
    current_sequence = last_sequence.copy()
    for _ in range(5):
        pred = model.predict(current_sequence, verbose=0)[0, 0]
        predictions.append(pred)
        
        # Update sequence for next prediction
        new_sequence = np.roll(current_sequence[0], -1)
        new_sequence[-1] = pred
        current_sequence = new_sequence.reshape(1, -1, 1)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten(), model, None

# Mock news data for demonstration
def get_mock_news(ticker, max_articles=10):
    """Generate mock news articles for demonstration"""
    sample_news = [
        {
            'title': f'{ticker} Reports Strong Q4 Earnings, Beats Expectations',
            'link': 'https://example.com/news1',
            'published': 'Mon, 15 Jan 2024 10:30:00 GMT',
            'summary': f'{ticker} exceeded analyst expectations with robust quarterly performance driven by strong product demand.'
        },
        {
            'title': f'Analysts Upgrade {ticker} Stock Rating Following Innovation Announcement',
            'link': 'https://example.com/news2', 
            'published': 'Sun, 14 Jan 2024 14:15:00 GMT',
            'summary': f'Wall Street analysts express optimism about {ticker}\'s future prospects following recent strategic announcements.'
        },
        {
            'title': f'{ticker} Faces Regulatory Scrutiny Over Market Practices',
            'link': 'https://example.com/news3',
            'published': 'Sat, 13 Jan 2024 09:45:00 GMT', 
            'summary': f'Regulatory authorities are examining {ticker}\'s business practices amid growing market competition concerns.'
        },
        {
            'title': f'Market Volatility Impacts {ticker} Trading Volume',
            'link': 'https://example.com/news4',
            'published': 'Fri, 12 Jan 2024 16:20:00 GMT',
            'summary': f'{ticker} shares experience increased volatility as broader market conditions remain uncertain.'
        },
        {
            'title': f'{ticker} Announces Partnership with Tech Industry Leader',
            'link': 'https://example.com/news5',
            'published': 'Thu, 11 Jan 2024 11:00:00 GMT',
            'summary': f'{ticker} enters strategic partnership expected to accelerate growth in key technology sectors.'
        },
        {
            'title': f'Economic Outlook Affects {ticker} Investor Sentiment',
            'link': 'https://example.com/news6',
            'published': 'Wed, 10 Jan 2024 13:30:00 GMT',
            'summary': f'Broader economic indicators influence investor confidence in {ticker} and similar market leaders.'
        }
    ]
    
    return sample_news[:max_articles]

# Fetch news
@st.cache_data(ttl=1800)
def fetch_news(ticker, max_articles=10):
    """Fetch news for a given ticker"""
    news_articles = []
    
    # Try Yahoo Finance RSS first
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:max_articles]:
            article = {
                'title': entry.title,
                'link': entry.link,
                'published': entry.published if hasattr(entry, 'published') else '',
                'summary': entry.summary if hasattr(entry, 'summary') else entry.title
            }
            news_articles.append(article)
    except:
        pass
    
    # Fallback to Google News RSS
    if len(news_articles) < 5:
        try:
            google_rss = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(google_rss)
            
            for entry in feed.entries[:max_articles-len(news_articles)]:
                article = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else '',
                    'summary': entry.summary if hasattr(entry, 'summary') else entry.title
                }
                news_articles.append(article)
        except:
            pass
    
    # Use mock data if no real news is available
    if len(news_articles) == 0:
        news_articles = get_mock_news(ticker, max_articles)
    
    return news_articles

# Analyze sentiment
def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity
    except:
        return "neutral", 0.0

# Create price chart
def create_price_chart(data, predictions_dict, ticker):
    """Create interactive price chart with predictions"""
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{ticker} Price Chart with Forecasts"])
    
    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2196F3', width=2)
        )
    )
    
    # Add prediction lines
    colors = {'LSTM': '#FF5722', 'GRU': '#4CAF50', 'Transformer': '#9C27B0'}
    
    # Create future dates (next 5 trading days)
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=5)
    
    for model_name, predictions in predictions_dict.items():
        if predictions is not None and len(predictions) == 5:
            # Connect last historical point to first prediction
            x_values = [last_date] + list(future_dates)
            y_values = [data['Close'].iloc[-1]] + list(predictions)
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors.get(model_name, '#666666'), width=2, dash='dash'),
                    marker=dict(size=6)
                )
            )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def main():
    """Main Streamlit app"""
    load_css()
    init_session_state()
    
    # Header with dark mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>📈 Ticker Insights</h1>", unsafe_allow_html=True)
    
    # Demo mode notice
    st.info("🚀 **Demo Mode**: Using sample data for demonstration. In production, this would connect to live Yahoo Finance data.")
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Ticker input section
    st.markdown("### Enter Stock Ticker")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            placeholder="e.g., AAPL, TSLA, GOOGL",
            label_visibility="collapsed"
        )
        
        # Validate ticker
        validation_placeholder = st.empty()
        if ticker_input and len(ticker_input) >= 2:
            # Show suggestions first
            suggestions = get_ticker_suggestions(ticker_input)
            if suggestions and ticker_input.upper() not in suggestions:
                st.info(f"💡 Suggestions: {', '.join(suggestions[:5])}")
            
            # Only validate if user has stopped typing (simple approach)
            if len(ticker_input) > 3:  # Only show error for longer inputs
                is_valid, error_msg = validate_ticker(ticker_input.upper())
                if not is_valid:
                    validation_placeholder.markdown(f'<div class="error-message">{error_msg}</div>', unsafe_allow_html=True)
    
    with col2:
        get_insights = st.button("Get Insights", type="primary", use_container_width=True)
    
    # Main content area
    if get_insights and ticker_input:
        ticker = ticker_input.upper()
        is_valid, error_msg = validate_ticker(ticker)
        
        if not is_valid:
            st.error(error_msg)
            return
        
        # Timeframe selector
        st.markdown("### Chart Timeframe")
        timeframes = {
            "1W": "1wk", "1M": "1mo", "3M": "3mo", "6M": "6mo",
            "1Y": "1y", "3Y": "3y", "5Y": "5y", "Max": "max"
        }
        
        selected_timeframe = st.selectbox(
            "Select timeframe:",
            options=list(timeframes.keys()),
            index=4  # Default to 1Y
        )
        
        # Custom date range option
        if selected_timeframe == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            period = None
        else:
            period = timeframes[selected_timeframe]
        
        # Fetch data and create predictions
        with st.spinner("Fetching data and generating forecasts..."):
            # Fetch stock data
            data, error = fetch_stock_data_no_cache(ticker, period)  # Use non-cached version directly
            if error and "Error generating demo data" in error:
                st.error(error)
                return
            elif error:
                st.info(error)  # Show info about using demo data
            
            # Generate predictions from all models
            predictions_dict = {}
            model_names = ["LSTM", "GRU", "Transformer"]
            
            for model_name in model_names:
                with st.spinner(f"Training {model_name} model..."):
                    predictions, model, error = train_and_predict(model_name, data, ticker)
                    if error:
                        st.warning(f"{model_name} model error: {error}")
                        predictions_dict[model_name] = None
                    else:
                        predictions_dict[model_name] = predictions
            
            # Create and display chart
            fig = create_price_chart(data, predictions_dict, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction summary
            st.markdown("### 5-Day Price Forecasts")
            future_dates = pd.bdate_range(start=data.index[-1] + timedelta(days=1), periods=5)
            
            forecast_df = pd.DataFrame(index=future_dates)
            current_price = data['Close'].iloc[-1]
            
            for model_name, predictions in predictions_dict.items():
                if predictions is not None:
                    forecast_df[f"{model_name} Price"] = predictions
                    forecast_df[f"{model_name} Change (%)"] = ((predictions - current_price) / current_price * 100).round(2)
            
            if not forecast_df.empty:
                st.dataframe(forecast_df.round(2), use_container_width=True)
            
            # Fetch and display news
            st.markdown("### Recent News & Sentiment")
            with st.spinner("Fetching latest news..."):
                news_articles = fetch_news(ticker)
                
                if news_articles:
                    for article in news_articles:
                        sentiment, polarity = analyze_sentiment(article['title'] + ' ' + article.get('summary', ''))
                        
                        # Create colored card based on sentiment
                        if sentiment == "positive":
                            card_class = "positive"
                            emoji = "📈"
                        elif sentiment == "negative":
                            card_class = "negative"
                            emoji = "📉"
                        else:
                            card_class = "neutral"
                            emoji = "📰"
                        
                        st.markdown(f"""
                        <div class="news-card {card_class}">
                            <h4>{emoji} {article['title']}</h4>
                            <p><strong>Sentiment:</strong> {sentiment.title()} ({polarity:.2f})</p>
                            <p><small>Published: {article.get('published', 'N/A')}</small></p>
                            <a href="{article['link']}" target="_blank">Read more →</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent news found for this ticker.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()