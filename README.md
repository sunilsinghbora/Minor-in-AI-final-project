# Stock AI Analyzer

A comprehensive stock analysis application with AI-powered predictions and sentiment analysis. The app provides an intuitive interface for searching ticker symbols, viewing interactive price charts, running AI models, and analyzing news sentiment.

## Features

### 🎯 Core Functionality
- **Ticker Symbol Autocomplete**: Input box with real-time suggestions
- **Interactive Price Charts**: Multiple time ranges (5D to MAX) with Plotly.js
- **AI Model Selection**: Choose from LSTM and GRU models
- **Real-time Progress**: Dynamic progress indicators during model training
- **Sentiment Analysis**: News-based sentiment scoring with circular progress bars
- **Expert Mode**: Advanced controls for model configuration and metrics

### 🎨 Design & UI
- **Single Dark Theme**: Modern, accessible dark mode for optimal readability
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Interactive Charts**: Color-coded lines (white for historical, orange/green for predictions)
- **Ticker Logo**: Company logo shown in analysis screen header
- **Watermark**: Subtle bottom-right watermark for project attribution
- **Modern UI**: Smooth transitions and hover effects

### 🤖 AI Models
- **LSTM**: Long Short-Term Memory for long-term dependencies
- **GRU**: Gated Recurrent Unit for efficient processing

## Project Structure

```
├── frontend/           # React TypeScript application
│   ├── src/
│   │   ├── components/
│   │   │   ├── TickerSearch.tsx    # Ticker search with autocomplete, preview
│   │   │   └── StockAnalysis.tsx   # Analysis screen with charts, logo, expert mode
│   │   ├── App.tsx                 # Main application component
│   │   └── index.css               # Custom styling
│   └── package.json
├── backend/            # Python Flask API
│   ├── app.py         # Main Flask application
│   ├── requirements.txt
│   └── venv/          # Virtual environment (ignored in git)
└── README.md
```

## Getting Started

### Prerequisites
- Node.js (v14 or later)
- Python 3.8+
- npm or yarn

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Flask server:
```bash
python app.py
```

The backend API will be available at `http://localhost:5000`

## Usage

### First Screen - Ticker Search
1. **Type a ticker symbol** (e.g., AAPL, MSFT, GOOGL)
2. **Select from autocomplete suggestions** that appear as you type
3. **Click "Analyze Stock"** to proceed to the analysis screen

### Second Screen - Stock Analysis
1. **Select time range** using the buttons (5D, 1W, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, MAX)
2. **Choose an AI model** from the dropdown (LSTM, GRU)
3. **Expert Mode**: Toggle for advanced controls (training range, epochs, window, dropout, horizon)
4. **Click "Run Model"** to start prediction
5. **Watch the progress** circle fill up dynamically
6. **View predictions** overlaid on the historical chart
7. **Check sentiment score** in the circular progress indicator
8. **View model metrics and used config** (Expert Mode)

### Key Features in Action
- **Autocomplete**: Type "A" to see all tickers starting with A
- **Navigation**: Use the back button to return to ticker search
- **Ticker Logo**: Company logo shown in analysis header
- **Watermark**: "IIT Ropar, Minor In AI Project submission by Sunil Singh Bora" in bottom-right
- **Predictions**: Short-horizon forecasts with 95% training, 5% testing data
- **Sentiment**: Score out of 100 based on Yahoo Finance & Google News headlines
- **Responsive Design**: Works on all screen sizes

## API Endpoints

### Backend API (localhost:5000)
- `GET /api/tickers/search?q={query}` - Search ticker symbols (autocomplete)
- `GET /api/stock/{ticker}/data?range={timeRange}` - Get historical stock data
- `POST /api/stock/{ticker}/predict` - Generate predictions (LSTM/GRU, expert mode supported)
- `GET /api/stock/{ticker}/sentiment` - Get sentiment analysis (Yahoo Finance & Google News)
- `GET /api/progress/{key}` - Get model training progress
- `GET /api/stock/{ticker}/csv?range={timeRange}` - Download historical data as CSV
- `GET /api/health` - Health check

## Technologies Used

### Frontend
- **React.js** with TypeScript for type safety
- **Plotly.js** for interactive charts
- **Custom CSS** for single dark theme
- **Dynamic logo loading** for ticker/company
- **Watermark** for project attribution
- **React Hooks** for state management

### Backend
- **Flask** for REST API
- **Flask-CORS** for cross-origin requests
- **Pandas & NumPy** for data processing
- **TensorFlow/Keras** for LSTM/GRU models
- **Yahoo Finance API** for real price and company data
- **Google News & Yahoo Finance RSS** for sentiment analysis

### Code Documentation
- All backend and frontend functions/methods are documented with comments and docstrings for clarity.

### Future Enhancements
- Additional ML models (CNN, TFT, etc.)
- WebSocket support for real-time updates
- User authentication and portfolio tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.