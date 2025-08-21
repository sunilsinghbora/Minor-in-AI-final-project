import React, { useState, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

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
  { label: 'LSTM (Long Short-Term Memory)', value: 'lstm' },
  { label: 'GRU (Gated Recurrent Unit)', value: 'gru' }
];

// Expert Mode: We now expose two controls instead of pre-defined profiles:
// - History range dropdown
// - Epochs dropdown

// (removed) mock data generator; now using backend API

// Use relative base by default so CRA proxy works (especially in Codespaces/containers)
const API_BASE = (process.env.REACT_APP_API_BASE as string) || '';

const StockAnalysis: React.FC<StockAnalysisProps> = ({ ticker, onBack }) => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('1m');
  const [selectedModel, setSelectedModel] = useState('lstm');
  // Expert Mode selections
  const [historyRange, setHistoryRange] = useState('1y');
  const [epochChoice, setEpochChoice] = useState<number>(8);
  const [dropout, setDropout] = useState<number>(0.3);
  const [windowLen, setWindowLen] = useState<number>(60);
  const [isModelRunning, setIsModelRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressEpoch, setProgressEpoch] = useState<number>(0);
  const [progressTotal, setProgressTotal] = useState<number>(0);
  // removed progressKey local state (key is kept in closure during a run)
  const [progressStatus, setProgressStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const progressTimerRef = useRef<any>(null);
  const [showPrediction, setShowPrediction] = useState(false);
  const [expertMode, setExpertMode] = useState(false);
  const [stockData, setStockData] = useState<any[]>([]);
  const [companyName, setCompanyName] = useState<string>('');
  const [predictionData, setPredictionData] = useState<any[]>([]);
  const [fittedData, setFittedData] = useState<Array<{ date: string; price: number }>>([]);
  const [testBoundaryIndex, setTestBoundaryIndex] = useState<number | null>(null);
  const [testBoundaryDate, setTestBoundaryDate] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any | null>(null);
  const [usedConfig, setUsedConfig] = useState<any | null>(null);
  const [forecastHorizon, setForecastHorizon] = useState<number>(5);
  const [sentimentScore, setSentimentScore] = useState<number>(0);
  const [loadingData, setLoadingData] = useState<boolean>(false);
  const [loadingSentiment, setLoadingSentiment] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // Whether to clip prediction (fitted) traces to the selected visible range
  const [clipPredToRange, setClipPredToRange] = useState<boolean>(true);
  // Fit line now always shows across full history; no scope toggle
  // Warm-up removed
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);
  const [showRetryButton, setShowRetryButton] = useState(false);
  const retryRef = useRef(0);
  const pingBackend = async () => {
    try {
      setBackendHealthy(null);
      const res = await axios.get(`${API_BASE}/api/health`, { timeout: 4000 });
      setBackendHealthy(Boolean(res.data?.status === 'healthy'));
    } catch {
      setBackendHealthy(false);
    }
  };

  const reloadStockData = async () => {
    setLoadingData(true);
    setError(null);
    try {
      const res = await axios.get(`${API_BASE}/api/stock/${encodeURIComponent(ticker)}/data`, {
        params: { range: selectedTimeRange }
      });
      const arr = res.data?.data || [];
      setStockData(arr);
      if (!arr.length) setError('No data returned. Try a different range or ticker.');
      setCompanyName(res.data?.companyName || ticker);
    } catch (e) {
      setError('Failed to load stock data. Backend may be unreachable.');
    } finally {
      setLoadingData(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingData(true);
      setError(null);
      try {
  const res = await axios.get(`${API_BASE}/api/stock/${encodeURIComponent(ticker)}/data`, {
          params: { range: selectedTimeRange }
        });
        if (cancelled) return;
  const arr = res.data?.data || [];
  setStockData(arr);
  if (!arr.length) setError('No data returned. Try a different range or ticker.');
        setCompanyName(res.data?.companyName || ticker);
      } catch (e: any) {
  if (!cancelled) setError('Failed to load stock data. Backend may be unreachable.');
      } finally {
        if (!cancelled) setLoadingData(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [selectedTimeRange, ticker]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingSentiment(true);
      try {
        const res = await axios.get(`${API_BASE}/api/stock/${encodeURIComponent(ticker)}/sentiment`);
        if (cancelled) return;
        setSentimentScore(res.data?.score ?? 50);
      } catch (e) {
        if (!cancelled) setSentimentScore(50);
      } finally {
        if (!cancelled) setLoadingSentiment(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [ticker]);

  // Warm-up removed

  // Backend health check banner
  useEffect(() => {
    let cancelled = false;
    let timer: any;
    const ping = async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/health`, { timeout: 4000 });
        const ok = Boolean(res.data?.status === 'healthy');
        if (cancelled) return;
        setBackendHealthy(ok);
        if (ok) {
          retryRef.current = 0;
          setShowRetryButton(false);
          // If we don't have data yet, reload it now that backend is up
          if (!stockData.length && !loadingData) {
            reloadStockData();
          }
        }
      } catch {
        if (cancelled) return;
        setBackendHealthy(false);
        if (retryRef.current < 3) {
          retryRef.current += 1;
          timer = setTimeout(ping, 2000);
          return;
        } else {
          setShowRetryButton(true);
        }
      } finally {
        if (!cancelled && !timer) timer = setTimeout(ping, 10000);
      }
    };
    ping();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  const handleRunModel = async () => {
    setIsModelRunning(true);
    setProgress(0);
    setProgressStatus('running');
    // create a unique progress key so backend can report epoch progress
  const key = `p-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
    setShowPrediction(false);
    setMetrics(null);
  setFittedData([]);
    // Start polling backend for real training progress
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    progressTimerRef.current = setInterval(async () => {
      if (!key) return;
      try {
        const r = await axios.get(`${API_BASE}/api/progress/${encodeURIComponent(key)}`);
        const info = r.data || {};
        if (typeof info.percent === 'number') setProgress(Math.max(0, Math.min(100, Math.round(info.percent))));
  if (typeof info.current === 'number') setProgressEpoch(Number(info.current));
  if (typeof info.total === 'number') setProgressTotal(Number(info.total));
        if (info.status === 'done') {
          setProgressStatus('done');
          clearInterval(progressTimerRef.current);
          progressTimerRef.current = null;
        } else if (info.status === 'error') {
          setProgressStatus('error');
          clearInterval(progressTimerRef.current);
          progressTimerRef.current = null;
        } else {
          setProgressStatus('running');
        }
      } catch {
        // ignore transient errors; keep polling briefly
      }
    }, 800);
    try {
      // If expert mode is off, force max range and 20 epochs; else use selected history/epoch
  const cfg = expertMode ? { train_range: historyRange, epochs: epochChoice, window: windowLen } : { train_range: 'max', epochs: 20, window: 60 };
  const res = await axios.post(`${API_BASE}/api/stock/${encodeURIComponent(ticker)}/predict`, {
        model: selectedModel,
        train_range: cfg.train_range,
        epochs: cfg.epochs,
        horizon: forecastHorizon,
        expertMode: expertMode,
        dropout: expertMode ? dropout : undefined,
  window: cfg.window,
        progress_key: key
      });
  const preds = (res.data?.predictions || []) as Array<{ date: string; price: number }>;
      setPredictionData(preds);
  setFittedData((res.data?.fitted || []) as Array<{ date: string; price: number }>);
  // Capture backend-provided test boundary index for coloring fitted series
  const tBoundary = res.data?.test_boundary?.index;
  if (typeof tBoundary === 'number' && tBoundary >= 0) setTestBoundaryIndex(Number(tBoundary)); else setTestBoundaryIndex(null);
  const tBoundaryDate = res.data?.test_boundary?.date;
  setTestBoundaryDate(typeof tBoundaryDate === 'string' ? tBoundaryDate : null);
      setMetrics(res.data?.metrics || null);
  setUsedConfig(res.data?.used || null);
      setShowPrediction(true);
  // Do not change the user's selected time range after prediction
      setProgress(100);
      setProgressStatus('done');
    } catch (e: any) {
      setShowPrediction(false);
      setProgressStatus('error');
      const msg = e?.response?.data?.error || 'Prediction failed. Please try a different range or later.';
      setError(msg);
  setFittedData([]);
    } finally {
      if (progressTimerRef.current) { clearInterval(progressTimerRef.current); progressTimerRef.current = null; }
      setIsModelRunning(false);
    }
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
          <h1 className="analysis-title">{ticker} {companyName ? `• ${companyName}` : ''} Analysis</h1>
        </div>
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
    <span style={{ color: '#ddd', fontSize: 14 }}>Expert Mode</span>
          <div
            onClick={() => setExpertMode(!expertMode)}
            role="switch"
            aria-checked={expertMode}
            style={{ cursor: 'pointer', width: 80, height: 32, borderRadius: 16, background: expertMode ? '#00c853' : '#555', display: 'flex', alignItems: 'center', padding: 4, boxShadow: 'inset 0 0 4px rgba(0,0,0,0.4)' }}
            title={`Expert Mode ${expertMode ? 'ON' : 'OFF'}`}
          >
            <div style={{ width: 24, height: 24, borderRadius: '50%', background: '#fff', transform: expertMode ? 'translateX(48px)' : 'translateX(0px)', transition: 'transform 0.2s ease' }} />
          </div>
        </div>
      </div>

      {(backendHealthy === false) && (
        <div className="error-message" style={{ marginBottom: '1rem', background: '#552', border: '1px solid #a00', padding: '8px 12px', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
          <span>
            Backend is unreachable. Ensure the server is running on port 5000.
            {!showRetryButton && (
              <span style={{ marginLeft: 10, color: '#ddd' }}>Auto-retrying ({retryRef.current}/3)…</span>
            )}
          </span>
          {showRetryButton && (
            <button onClick={() => { pingBackend(); reloadStockData(); }} style={{ background: '#a00', color: '#fff', border: 'none', padding: '6px 10px', borderRadius: 4, cursor: 'pointer' }}>
              Retry
            </button>
          )}
        </div>
      )}
      {error && (
        <div className="error-message" style={{ marginBottom: '1rem' }}>{error}</div>
      )}

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
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            id="clipPredToRange"
            type="checkbox"
            checked={clipPredToRange}
            onChange={(e) => setClipPredToRange(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <label htmlFor="clipPredToRange" style={{ color: '#ddd', cursor: 'pointer' }}>
            Clip predictions to selected range
          </label>
        </div>
      </div>

      {/* Stock Price Chart */}
      <div className="chart-container">
        <h2 className="chart-title">Price Chart</h2>
        <Plot
          data={(() => {
            const historyColor = '#ffffff';
            const trainFitColor = '#ff8000';
            const testFitColor = '#00c853';
            const futureColor = '#00c853';
            // Ensure actual series always derives from loaded stockData
            const xs = (stockData || []).map((d: any) => d?.date).filter(Boolean);
            const ys = (stockData || []).map((d: any) => d?.price).filter((v: any) => v !== null && v !== undefined);
            const series: any[] = [];
            // Always show the actual historical data as a single white line
            if (xs.length && ys.length && xs.length === ys.length) {
              series.push({ x: xs, y: ys, type: 'scatter' as const, mode: 'lines' as const, name: 'Actual', line: { color: historyColor, width: 2 } });
            }
            // Show fitted line; optionally clip to the currently visible range, then split into train (orange) and test (green)
            if (showPrediction && fittedData.length) {
              // Clip fitted predictions to date boundaries of the visible history instead of exact date matches,
              // so we still show daily fitted values when the history is weekly/monthly aggregated.
              let base = fittedData;
              if (clipPredToRange && xs.length) {
                const xMin = xs[0];
                const xMax = xs[xs.length - 1];
                base = fittedData.filter(d => d.date >= xMin && d.date <= xMax);
              }
              if (base.length) {
                // Prefer splitting by boundary date from backend; fallback to approximate index
                let splitPos = Math.floor(base.length * 0.95);
                if (testBoundaryDate) {
                  const idx = base.findIndex(d => d.date >= testBoundaryDate);
                  if (idx >= 0) splitPos = idx;
                } else if (typeof testBoundaryIndex === 'number' && testBoundaryIndex >= 0) {
                  splitPos = Math.max(0, Math.min(testBoundaryIndex, base.length));
                }
                const trainSeg = base.slice(0, Math.max(0, Math.min(splitPos, base.length)));
                const testSeg = base.slice(Math.max(0, Math.min(splitPos, base.length)));
                if (trainSeg.length) {
                  series.push({ x: trainSeg.map(d => d.date), y: trainSeg.map(d => d.price), type: 'scatter' as const, mode: 'lines' as const, name: 'Prediction (train)', line: { color: trainFitColor, width: 2 } });
                }
                if (testSeg.length) {
                  series.push({ x: testSeg.map(d => d.date), y: testSeg.map(d => d.price), type: 'scatter' as const, mode: 'lines' as const, name: 'Prediction (test)', line: { color: testFitColor, width: 2 } });
                }
              }
            }
            // Future forecast (dotted green)
            if (showPrediction && predictionData.length) {
              const px = predictionData.map(d => d.date);
              const py = predictionData.map(d => d.price);
              if (px.length && py.length) {
                series.push({ x: px, y: py, type: 'scatter' as const, mode: 'lines' as const, name: 'Prediction (future)', line: { color: futureColor, width: 2, dash: 'dot' as const } });
              }
            }
            return series;
          })()}
      layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#ffffff' },
            xaxis: { 
              gridcolor: '#333333',
        title: { text: 'Date' }
            },
            yaxis: { 
              gridcolor: '#333333',
        title: { text: 'Price ($)' }
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
        {loadingData && (
          <div style={{ marginTop: '0.5rem', color: '#aaa' }}>Loading data…</div>
        )}
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

          {expertMode && (
            <>
              <h3 className="control-title" style={{ marginTop: '1rem' }}>Training Data Range</h3>
              <select
                value={historyRange}
                onChange={(e) => setHistoryRange(e.target.value)}
                className="model-select"
                disabled={isModelRunning}
              >
                <option value="5y">5 Years</option>
                <option value="10y">10 Years</option>
                <option value="15y">15 Years</option>
                <option value="20y">20 Years</option>
                <option value="max">Entire History</option>
              </select>
              <h3 className="control-title" style={{ marginTop: '1rem' }}>Epochs</h3>
              <select
                value={epochChoice}
                onChange={(e) => setEpochChoice(parseInt(e.target.value, 10))}
                className="model-select"
                disabled={isModelRunning}
              >
                {[2,4,6,8,10,12,15,20,25,30,35,40,45,50].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
              <h3 className="control-title" style={{ marginTop: '1rem' }}>Forecast Horizon (days)</h3>
              <select
                value={forecastHorizon}
                onChange={(e) => setForecastHorizon(parseInt(e.target.value, 10))}
                className="model-select"
                disabled={isModelRunning}
              >
                {[1,5,10,20,30,60,90].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
              <h3 className="control-title" style={{ marginTop: '1rem' }}>Window (days)</h3>
              <select
                value={windowLen}
                onChange={(e) => setWindowLen(parseInt(e.target.value, 10))}
                className="model-select"
                disabled={isModelRunning}
              >
                {[20,30,40,50,60,75,90,120].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
              <h3 className="control-title" style={{ marginTop: '1rem' }}>Dropout</h3>
              <select
                value={dropout}
                onChange={(e) => setDropout(parseFloat(e.target.value))}
                className="model-select"
                disabled={isModelRunning}
              >
                {[0,0.1,0.2,0.3,0.4,0.5].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </>
          )}
          
          <button
            onClick={handleRunModel}
            disabled={isModelRunning}
            className="run-model-button"
          >
            {isModelRunning ? 'Running Model...' : 'Run Model'}
          </button>

          {!expertMode && (
            <div style={{ marginTop: '0.5rem', color: '#bbb', fontSize: '0.9rem' }}>
              Using all available history with ~95% train / 5% test (window=60, epochs=20, batch=32). Forecasts default to the next 5 trading days.
            </div>
          )}

          {/* Progress Circle */}
      {isModelRunning && (
            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
              <CircularProgress 
  score={progress}
  label={progressStatus === 'running' ? `${progressEpoch}/${progressTotal}` : progressStatus === 'done' ? 'Done' : progressStatus === 'error' ? 'Error' : `${Math.round(progress)}%`}
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
              label={loadingSentiment ? 'Loading…' : 'Sentiment Score'}
              color={sentimentScore >= 50 ? '#00ff00' : '#ff0000'}
            />
          </div>
          <p className="sentiment-description">
            Based on Yahoo Finance and Google News headlines (recent)
          </p>
        </div>

        {/* Model Performance (Dev Mode) */}
  {expertMode && (
              <div className="control-panel">
            <h3 className="control-title">Model Metrics</h3>
            <div className="metrics-list">
              <div className="metric-row">
                <span>RMSE:</span>
                <span>{metrics?.rmse ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>MAE:</span>
                <span>{metrics?.mae ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>R²:</span>
                <span>{metrics?.r2 ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>Training Acc:</span>
                <span>—</span>
              </div>
              <div className="metric-row">
                <span>Val Acc:</span>
                <span>—</span>
              </div>
              <div className="metric-row">
                <span>Epochs:</span>
                <span>{metrics?.epochs ?? '—'}</span>
              </div>
            </div>
          </div>
        )}
        {/* Used Config Panel */}
        {showPrediction && (
          <div className="control-panel">
            <h3 className="control-title">Used Config</h3>
            <div className="metrics-list">
              <div className="metric-row">
                <span>Window:</span>
                <span>{usedConfig?.window ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>Batch size:</span>
                <span>{usedConfig?.batch_size ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>Dropout:</span>
                <span>{usedConfig?.dropout ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>Horizon:</span>
                <span>{usedConfig?.horizon ?? forecastHorizon}</span>
              </div>
              <div className="metric-row">
                <span>Train range:</span>
                <span>{usedConfig?.train_range ?? (expertMode ? historyRange : 'max')}</span>
              </div>
              <div className="metric-row">
                <span>Features:</span>
                <span>{usedConfig?.features ?? '—'}</span>
              </div>
              <div className="metric-row">
                <span>Test split:</span>
                <span>{usedConfig?.test_split ?? '—'}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Prediction Results */}
      {showPrediction && (
        <div className="prediction-results">
      <h3 className="control-title">Prediction Results (Next {forecastHorizon} Days)</h3>
          <div className="prediction-grid">
            <div>
              <div className="prediction-value">
                ${predictionData[Math.min(0, predictionData.length - 1)]?.price?.toFixed(2) || 'N/A'}
              </div>
              <div className="prediction-label">Next Day</div>
            </div>
            <div>
              <div className="prediction-value">
                ${predictionData[Math.min(4, predictionData.length - 1)]?.price?.toFixed(2) || 'N/A'}
              </div>
              <div className="prediction-label">1 Week</div>
            </div>
            <div>
              <div className="prediction-value">
                ${predictionData[Math.min(forecastHorizon - 1, predictionData.length - 1)]?.price?.toFixed(2) || 'N/A'}
              </div>
              <div className="prediction-label">Horizon End</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StockAnalysis;