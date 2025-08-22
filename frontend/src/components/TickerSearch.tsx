import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface TickerSearchProps {
  // Parent supplies a callback to open the analysis screen for a symbol
  onTickerSelect: (ticker: string) => void;
}

// Use relative base by default so CRA proxy works (especially in Codespaces/containers)
const API_BASE = (process.env.REACT_APP_API_BASE as string) || '';
// Local fallback if API is unreachable
const LOCAL_FALLBACK_TICKERS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'ADBE',
  'ORCL', 'CRM', 'INTC', 'CSCO', 'IBM', 'UBER', 'LYFT', 'SNAP', 'SPOT',
  'BABA', 'JD', 'NTES', 'BILI', 'PDD', 'TME', 'NIO', 'XPEV', 'LI',
  'BA', 'GE', 'F', 'GM', 'CAT', 'DE', 'MMM', 'HON', 'LMT',
  'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL'
];

const TickerSearch: React.FC<TickerSearchProps> = ({ onTickerSelect }) => {
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState<Array<{ symbol: string; name?: string }>>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [loading, setLoading] = useState(false);

  // Preview modal state
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewTicker, setPreviewTicker] = useState<string>('');
  const [previewRange] = useState<string>('max'); // load full available history for preview
  const [previewData, setPreviewData] = useState<Array<{ date: string; price: number; open?: number }>>([]);
  const [previewCompany, setPreviewCompany] = useState<string>('');
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    // Debounced autocomplete query; falls back to local list on error
    let cancelled = false;
    const q = inputValue.trim();
    if (!q) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/tickers/search`, { params: { q } });
        if (cancelled) return;
        const list = (res.data || []) as Array<{ symbol: string; name?: string }>;
        setSuggestions(list.slice(0, 8));
        setShowSuggestions(list.length > 0);
      } catch (e) {
        if (!cancelled) {
          const upper = q.toUpperCase();
          const local = LOCAL_FALLBACK_TICKERS
            .filter(s => s.startsWith(upper))
            .slice(0, 8)
            .map(s => ({ symbol: s, name: s }));
          setSuggestions(local);
          setShowSuggestions(local.length > 0);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 200); // debounce
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [inputValue]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value.toUpperCase());
  };

  const openPreview = async (ticker: string) => {
    // Open modal and load preview table (max history) for quick validation
    setPreviewTicker(ticker);
    setPreviewOpen(true);
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewData([]);
    try {
      const res = await axios.get(`${API_BASE}/api/stock/${encodeURIComponent(ticker)}/data`, {
        params: { range: previewRange }
      });
  const arr = (res.data?.data || []) as Array<{ date: string; price: number; open?: number }>;
      setPreviewData(arr);
      setPreviewCompany(res.data?.companyName || ticker);
      if (!arr.length) setPreviewError('No data returned for preview. Try a different ticker or later.');
    } catch (e: any) {
      setPreviewError('Failed to load preview data. Please try again.');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSuggestionClick = (ticker: string) => {
    setInputValue(ticker);
    setShowSuggestions(false);
    openPreview(ticker);
  };

  const handleAnalyzeClick = () => {
    // Open preview for the typed ticker or suggestion
    const t = inputValue.trim();
    if (!t) return;
    const match = suggestions.find(s => s.symbol.toUpperCase() === t.toUpperCase());
    openPreview(match ? match.symbol : t.toUpperCase());
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleAnalyzeClick();
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
          {suggestions.map((sugg) => (
                    <div
            key={sugg.symbol}
            onClick={() => handleSuggestionClick(sugg.symbol)}
                      className="suggestion-item"
                    >
            <div className="suggestion-ticker">{sugg.symbol}</div>
            <div className="suggestion-label">{sugg.name || 'Stock'}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyzeClick}
              disabled={!inputValue.trim()}
              className="analyze-button"
            >
              {loading ? 'Searching...' : 'Analyze Stock'}
            </button>
          </div>

          {/* Invalid ticker message */}
          {inputValue.trim() && !showSuggestions && (
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

      {/* Preview Modal */}
      {previewOpen && (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999
          }}
          onClick={(e) => {
            // close if clicking on backdrop
            if (e.target === e.currentTarget) setPreviewOpen(false);
          }}
        >
          <div style={{
            background: '#1e1e1e', color: '#fff', width: '90%', maxWidth: 900,
            borderRadius: 8, boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
            border: '1px solid #333', maxHeight: '85vh', display: 'flex', flexDirection: 'column'
          }}>
            <div style={{ padding: '14px 16px', borderBottom: '1px solid #333', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: 18, fontWeight: 600 }}>Confirm Data for {previewTicker}</div>
                <div style={{ fontSize: 12, color: '#aaa' }}>{previewCompany || ''} • Preview range: {previewRange.toUpperCase()}</div>
              </div>
              <button onClick={() => setPreviewOpen(false)} style={{ background: 'transparent', border: 'none', color: '#bbb', fontSize: 20, cursor: 'pointer' }}>×</button>
            </div>

            <div style={{ padding: 16 }}>
              {previewLoading && <div style={{ color: '#ccc' }}>Loading preview…</div>}
              {previewError && <div className="error-message" style={{ marginBottom: 8 }}>{previewError}</div>}
              {!previewLoading && !previewError && (
                <div style={{ border: '1px solid #333', borderRadius: 6, overflow: 'hidden' }}>
                  <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead style={{ position: 'sticky', top: 0, background: '#262626', zIndex: 1 }}>
                        <tr>
                          <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '1px solid #333' }}>Date</th>
                          <th style={{ textAlign: 'right', padding: '8px 10px', borderBottom: '1px solid #333' }}>Open</th>
                          <th style={{ textAlign: 'right', padding: '8px 10px', borderBottom: '1px solid #333' }}>Close</th>
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.map((row: any, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid #2a2a2a' }}>
                            <td style={{ padding: '8px 10px' }}>{row.date}</td>
                            <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                              {typeof row.open === 'number' ? `$${row.open.toFixed(2)}` : '—'}
                            </td>
                            <td style={{ padding: '8px 10px', textAlign: 'right' }}>${row.price.toFixed(2)}</td>
                          </tr>
                        ))}
                        {(!previewData || previewData.length === 0) && (
                          <tr>
                            <td colSpan={3} style={{ padding: '12px 10px', color: '#999' }}>No data</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '12px 16px', borderTop: '1px solid #333' }}>
              <button
                onClick={() => setPreviewOpen(false)}
                style={{ background: '#333', color: '#fff', border: '1px solid #555', padding: '8px 12px', borderRadius: 6, cursor: 'pointer' }}
              >
                Cancel
              </button>
              <button
                onClick={() => { setPreviewOpen(false); onTickerSelect(previewTicker); }}
                disabled={previewLoading || (!!previewError && (!previewData || previewData.length === 0))}
                style={{ background: '#0a84ff', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: 6, cursor: 'pointer', opacity: (previewLoading ? 0.7 : 1) }}
              >
                Proceed to Analysis
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TickerSearch;