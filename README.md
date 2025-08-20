# Stock AI Analyzer

A comprehensive stock analysis application with AI-powered predictions and sentiment analysis, featuring a ChatGPT-style interface.

## Features

### 🎯 Core Functionality
- **Ticker Symbol Autocomplete**: ChatGPT-style input with real-time suggestions
- **Interactive Price Charts**: Multiple time ranges (5D to MAX) with Plotly.js
- **AI Model Selection**: Choose from RNN, LSTM, GRU, CNN, and TFT models
- **Real-time Progress**: Dynamic progress indicators during model training
- **Sentiment Analysis**: News-based sentiment scoring with circular progress bars
- **Developer Mode**: Technical metrics and model performance indicators

### 🎨 Design
- **Black/White Theme**: Clean, contrasting design for optimal readability
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Interactive Charts**: Color-coded lines (green for historical, orange for predictions)
- **Modern UI**: Smooth transitions and hover effects

### 🤖 AI Models
- **RNN**: Recurrent Neural Network for sequential data
- **LSTM**: Long Short-Term Memory for long-term dependencies
- **GRU**: Gated Recurrent Unit for efficient processing
- **CNN**: Convolutional Neural Network for pattern recognition
- **TFT**: Temporal Fusion Transformer for advanced time series

## Project Structure

```
├── frontend/           # React TypeScript application
│   ├── src/
│   │   ├── components/
│   │   │   ├── TickerSearch.tsx    # First screen with autocomplete
│   │   │   └── StockAnalysis.tsx   # Analysis screen with charts
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
2. **Choose an AI model** from the dropdown (RNN, LSTM, GRU, CNN, TFT)
3. **Click "Run Model"** to start prediction
4. **Watch the progress** circle fill up dynamically
5. **View predictions** overlaid on the historical chart
6. **Check sentiment score** in the circular progress indicator
7. **Toggle Dev Mode** to see technical metrics

### Key Features in Action
- **Autocomplete**: Type "A" to see all tickers starting with A
- **Navigation**: Use the back button to return to ticker search
- **Predictions**: 3-month forecasts with 90% training, 10% testing data
- **Sentiment**: Score out of 100 based on recent news analysis
- **Responsive Design**: Works on all screen sizes

## API Endpoints

### Backend API (localhost:5000)
- `GET /api/tickers/search?q={query}` - Search ticker symbols
- `GET /api/stock/{ticker}/data?range={timeRange}` - Get stock data
- `POST /api/stock/{ticker}/predict` - Generate predictions
- `GET /api/stock/{ticker}/sentiment` - Get sentiment analysis
- `GET /api/health` - Health check

## Technologies Used

### Frontend
- **React.js** with TypeScript for type safety
- **Plotly.js** for interactive charts
- **Custom CSS** for black/white theme
- **React Hooks** for state management

### Backend
- **Flask** for REST API
- **Flask-CORS** for cross-origin requests
- **Pandas & NumPy** for data processing
- **Mock data generation** for demonstration

### Future Enhancements
- Real stock data integration (Yahoo Finance API)
- Actual ML model implementations with TensorFlow
- News API integration for sentiment analysis
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