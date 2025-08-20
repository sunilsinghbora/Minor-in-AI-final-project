# Minimalistic Ticker Insights App

A minimalist Streamlit application that provides stock market insights with AI-powered price forecasts and sentiment analysis.

## Features

- **Clean GitHub-style UI** with dark mode toggle
- **Live ticker validation** with suggestions
- **Interactive price charts** with multiple timeframe options (1W, 1M, 3M, 6M, 1Y, 3Y, 5Y, Max)
- **AI-powered forecasts** using three different models:
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - Transformer (Multi-Head Attention)
- **5-day price predictions** with visual overlay
- **News sentiment analysis** with color-coded cards:
  - 📈 Green: Positive sentiment
  - 📉 Red: Negative sentiment
  - 📰 Gray: Neutral sentiment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sunilsinghbora/Minor-in-AI-final-project.git
cd Minor-in-AI-final-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Enter a stock ticker symbol (e.g., AAPL, TSLA, GOOGL)

4. Click "Get Insights" to view:
   - Historical price chart
   - AI model forecasts for next 5 trading days
   - Recent news with sentiment analysis

## Technical Details

- **Data Source**: Yahoo Finance (no API key required)
- **News Sources**: Yahoo Finance RSS + Google News RSS fallback
- **ML Models**: Lightweight models with quick training (10 epochs) for demo purposes
- **Sentiment Analysis**: TextBlob for natural language processing
- **Charts**: Interactive Plotly visualizations

## Features in Detail

### Ticker Input
- Real-time validation
- Popular ticker suggestions
- Error handling for invalid symbols

### Price Charts
- Interactive Plotly charts
- Multiple timeframe selection
- Historical price data
- Forecast overlay lines

### AI Forecasts
- Three different model architectures
- 5-day ahead predictions
- Percentage change calculations
- Model comparison

### News & Sentiment
- Last 30 days of news articles
- Automated sentiment scoring
- Color-coded sentiment cards
- Direct links to articles

## Dependencies

- streamlit: Web app framework
- yfinance: Stock data retrieval
- tensorflow: Machine learning models
- plotly: Interactive charts
- textblob: Sentiment analysis
- feedparser: News RSS parsing
- pandas, numpy: Data manipulation
- scikit-learn: Data preprocessing

## Note

This is a demonstration application. The ML models use minimal training for responsiveness and should not be used for actual trading decisions.