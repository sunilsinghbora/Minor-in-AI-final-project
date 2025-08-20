import React, { useState, useEffect } from 'react';

// Mock ticker data for autocomplete (in a real app, this would come from an API)
const MOCK_TICKERS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'ADBE',
  'ORCL', 'CRM', 'INTC', 'CSCO', 'IBM', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SPOT',
  'BABA', 'JD', 'NTES', 'BILI', 'PDD', 'TME', 'DIDI', 'NIO', 'XPEV', 'LI',
  'BA', 'GE', 'F', 'GM', 'CAT', 'DE', 'MMM', 'HON', 'UTX', 'LMT',
  'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL'
];

interface TickerSearchProps {
  onTickerSelect: (ticker: string) => void;
}

const TickerSearch: React.FC<TickerSearchProps> = ({ onTickerSelect }) => {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  useEffect(() => {
    if (inputValue.trim() === '') {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    const filteredSuggestions = MOCK_TICKERS.filter(ticker =>
      ticker.toLowerCase().startsWith(inputValue.toLowerCase())
    ).slice(0, 8); // Show max 8 suggestions

    setSuggestions(filteredSuggestions);
    setShowSuggestions(filteredSuggestions.length > 0);
  }, [inputValue]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value.toUpperCase());
  };

  const handleSuggestionClick = (ticker: string) => {
    setInputValue(ticker);
    setShowSuggestions(false);
    onTickerSelect(ticker);
  };

  const handleAnalyzeClick = () => {
    if (inputValue.trim() && MOCK_TICKERS.includes(inputValue.trim())) {
      onTickerSelect(inputValue.trim());
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim() && MOCK_TICKERS.includes(inputValue.trim())) {
      onTickerSelect(inputValue.trim());
    }
  };

  return (
    <div className="search-screen">
      <div className="search-content">
        {/* Title */}
        <h1 className="main-title">
          Stock AI Analyzer
        </h1>
        <p className="subtitle">
          Enter a stock ticker symbol to analyze with AI models
        </p>

        {/* Search Container */}
        <div className="search-container">
          <div className="search-input-group">
            {/* Input Field */}
            <div style={{ position: 'relative', flex: '1' }}>
              <input
                type="text"
                value={inputValue}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder="Type a ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                className="search-input"
              />
              
              {/* Suggestions Dropdown */}
              {showSuggestions && (
                <div className="suggestions-dropdown">
                  {suggestions.map((ticker, index) => (
                    <div
                      key={ticker}
                      onClick={() => handleSuggestionClick(ticker)}
                      className="suggestion-item"
                    >
                      <div className="suggestion-ticker">{ticker}</div>
                      <div className="suggestion-label">Stock Symbol</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyzeClick}
              disabled={!inputValue.trim() || !MOCK_TICKERS.includes(inputValue.trim())}
              className="analyze-button"
            >
              Analyze Stock
            </button>
          </div>

          {/* Invalid ticker message */}
          {inputValue.trim() && !MOCK_TICKERS.includes(inputValue.trim()) && !showSuggestions && (
            <div className="error-message">
              Invalid ticker symbol. Please select from suggestions.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="footer-text">
          <p>Powered by AI | Real-time analysis with multiple time series models</p>
        </div>
      </div>
    </div>
  );
};

export default TickerSearch;