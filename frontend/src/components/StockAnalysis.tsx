import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

interface StockAnalysisProps {
  ticker: string;
  onBack: () => void;
}

const TIME_RANGES = [
  { label: '5D', value: '5d' },
  { label: '1W', value: '1w' },
  { label: '1M', value: '1m' },
  { label: '3M', value: '3m' },
  { label: '6M', value: '6m' },
  { label: '1Y', value: '1y' },
  { label: '2Y', value: '2y' },
  { label: '3Y', value: '3y' },
  { label: '5Y', value: '5y' },
  { label: 'MAX', value: 'max' }
];

const ML_MODELS = [
  { label: 'RNN (Recurrent Neural Network)', value: 'rnn' },
  { label: 'LSTM (Long Short-Term Memory)', value: 'lstm' },
  { label: 'GRU (Gated Recurrent Unit)', value: 'gru' },
  { label: 'CNN (Convolutional Neural Network)', value: 'cnn' },
  { label: 'TFT (Temporal Fusion Transformer)', value: 'tft' }
];

// Mock data generator for demonstration
const generateMockData = (days: number) => {
  const data = [];
  const startPrice = 150 + Math.random() * 50;
  let currentPrice = startPrice;
  
  for (let i = 0; i < days; i++) {
    const date = new Date();
    date.setDate(date.getDate() - (days - i));
    
    currentPrice += (Math.random() - 0.5) * 10;
    currentPrice = Math.max(currentPrice, 10); // Minimum price
    
    data.push({
      date: date.toISOString().split('T')[0],
      price: currentPrice
    });
  }
  
  return data;
};

const StockAnalysis: React.FC<StockAnalysisProps> = ({ ticker, onBack }) => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('1m');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [isModelRunning, setIsModelRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showPrediction, setShowPrediction] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [stockData, setStockData] = useState<any[]>([]);
  const [predictionData, setPredictionData] = useState<any[]>([]);
  const [sentimentScore] = useState(75); // Mock sentiment score

  useEffect(() => {
    // Generate mock data based on time range
    const daysMap: Record<string, number> = {
      '5d': 5, '1w': 7, '1m': 30, '3m': 90, '6m': 180,
      '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, 'max': 2555
    };
    
    const data = generateMockData(daysMap[selectedTimeRange] || 30);
    setStockData(data);
  }, [selectedTimeRange, ticker]);

  const handleRunModel = async () => {
    setIsModelRunning(true);
    setProgress(0);
    setShowPrediction(false);

    // Simulate model training progress
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          setIsModelRunning(false);
          setShowPrediction(true);
          
          // Generate prediction data (next 3 months)
          const lastPrice = stockData[stockData.length - 1]?.price || 150;
          const predictions = [];
          let currentPrice = lastPrice;
          
          for (let i = 1; i <= 90; i++) { // 3 months
            const date = new Date();
            date.setDate(date.getDate() + i);
            currentPrice += (Math.random() - 0.45) * 8; // Slight upward trend
            
            predictions.push({
              date: date.toISOString().split('T')[0],
              price: currentPrice
            });
          }
          setPredictionData(predictions);
          
          return 100;
        }
        return prev + Math.random() * 15 + 5;
      });
    }, 200);
  };

  const CircularProgress: React.FC<{ score: number; label: string; color: string }> = 
    ({ score, label, color }) => {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const strokeDasharray = circumference;
    const strokeDashoffset = circumference - (score / 100) * circumference;

    return (
      <div className="circular-progress">
        <div className="progress-circle">
          <svg width="128" height="128" style={{ transform: 'rotate(-90deg)' }} viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r={radius}
              stroke="#666666"
              strokeWidth="8"
              fill="transparent"
            />
            <circle
              cx="50"
              cy="50"
              r={radius}
              stroke={color}
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              style={{ transition: 'all 1s ease-out' }}
              strokeLinecap="round"
            />
          </svg>
          <div className="progress-text">{Math.round(score)}</div>
        </div>
        <span className="progress-label">{label}</span>
      </div>
    );
  };

  return (
    <div className="analysis-container">
      {/* Header */}
      <div className="analysis-header">
        <div className="header-left">
          <button onClick={onBack} className="back-button">
            ← Back
          </button>
          <h1 className="analysis-title">{ticker} Analysis</h1>
        </div>
        <button
          onClick={() => setDevMode(!devMode)}
          className="dev-mode-button"
        >
          Dev Mode {devMode ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Time Range Selector */}
      <div className="time-range-selector">
        <div className="time-range-buttons">
          {TIME_RANGES.map(range => (
            <button
              key={range.value}
              onClick={() => setSelectedTimeRange(range.value)}
              className={`time-range-button ${selectedTimeRange === range.value ? 'active' : ''}`}
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stock Price Chart */}
      <div className="chart-container">
        <h2 className="chart-title">Price Chart</h2>
        <Plot
          data={[
            {
              x: stockData.map(d => d.date),
              y: stockData.map(d => d.price),
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'Historical Price',
              line: { color: '#00ff00', width: 2 }
            },
            ...(showPrediction ? [{
              x: predictionData.map(d => d.date),
              y: predictionData.map(d => d.price),
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'Predicted Price',
              line: { color: '#ff8000', width: 2, dash: 'dash' as const }
            }] : [])
          ]}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#ffffff' },
            xaxis: { 
              gridcolor: '#333333',
              title: 'Date'
            },
            yaxis: { 
              gridcolor: '#333333',
              title: 'Price ($)'
            },
            legend: {
              x: 0,
              y: 1
            },
            margin: { t: 40, l: 60, r: 40, b: 60 }
          }}
          style={{ width: '100%', height: '400px' }}
          config={{ displayModeBar: false }}
        />
      </div>

      {/* Model Selection and Controls */}
      <div className="controls-grid">
        {/* Model Selection */}
        <div className="control-panel">
          <h3 className="control-title">AI Model Selection</h3>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="model-select"
            disabled={isModelRunning}
          >
            {ML_MODELS.map(model => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </select>
          
          <button
            onClick={handleRunModel}
            disabled={isModelRunning}
            className="run-model-button"
          >
            {isModelRunning ? 'Running Model...' : 'Run Model'}
          </button>

          {/* Progress Circle */}
          {isModelRunning && (
            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
              <CircularProgress 
                score={progress} 
                label={`${Math.round(progress)}%`}
                color="#00ff00"
              />
            </div>
          )}
        </div>

        {/* Sentiment Analysis */}
        <div className="control-panel">
          <h3 className="control-title">Sentiment Analysis</h3>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <CircularProgress
              score={sentimentScore}
              label="Sentiment Score"
              color={sentimentScore >= 50 ? '#00ff00' : '#ff0000'}
            />
          </div>
          <p className="sentiment-description">
            Based on news analysis from the last 3 months
          </p>
        </div>

        {/* Model Performance (Dev Mode) */}
        {devMode && (
          <div className="control-panel">
            <h3 className="control-title">Model Metrics</h3>
            <div className="metrics-list">
              <div className="metric-row">
                <span>RMSE:</span>
                <span>2.34</span>
              </div>
              <div className="metric-row">
                <span>MAE:</span>
                <span>1.87</span>
              </div>
              <div className="metric-row">
                <span>R²:</span>
                <span>0.89</span>
              </div>
              <div className="metric-row">
                <span>Training Acc:</span>
                <span>94.2%</span>
              </div>
              <div className="metric-row">
                <span>Val Acc:</span>
                <span>91.8%</span>
              </div>
              <div className="metric-row">
                <span>Epochs:</span>
                <span>50</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Prediction Results */}
      {showPrediction && (
        <div className="prediction-results">
          <h3 className="control-title">Prediction Results (Next 3 Months)</h3>
          <div className="prediction-grid">
            <div>
              <div className="prediction-value">
                ${predictionData[29]?.price?.toFixed(2) || 'N/A'}
              </div>
              <div className="prediction-label">1 Month Target</div>
            </div>
            <div>
              <div className="prediction-value">
                ${predictionData[59]?.price?.toFixed(2) || 'N/A'}
              </div>
              <div className="prediction-label">2 Month Target</div>
            </div>
            <div>
              <div className="prediction-value">
                ${predictionData[89]?.price?.toFixed(2) || 'N/A'}
              </div>
              <div className="prediction-label">3 Month Target</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StockAnalysis;