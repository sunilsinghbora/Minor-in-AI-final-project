import React, { useState } from 'react';
import './App.css';
import TickerSearch from './components/TickerSearch';
import StockAnalysis from './components/StockAnalysis';

function App() {
  const [selectedTicker, setSelectedTicker] = useState<string>('');
  const [currentScreen, setCurrentScreen] = useState<'search' | 'analysis'>('search');

  const handleTickerSelect = (ticker: string) => {
    setSelectedTicker(ticker);
    setCurrentScreen('analysis');
  };

  const handleBackToSearch = () => {
    setCurrentScreen('search');
    setSelectedTicker('');
  };

  return (
    <div className="app-container">
      {currentScreen === 'search' ? (
        <TickerSearch onTickerSelect={handleTickerSelect} />
      ) : (
        <StockAnalysis 
          ticker={selectedTicker} 
          onBack={handleBackToSearch}
        />
      )}
    </div>
  );
}

export default App;
