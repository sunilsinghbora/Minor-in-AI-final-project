import numpy as np
from typing import List, Tuple, Dict, Optional, Union
try:
    # Prefer scikit-learn's implementation if available
    from sklearn.preprocessing import StandardScaler as _SKStandardScaler  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

from .models import get_model_builder

# Default training window (trading days) and forecast horizon (days)
# 60 gives ~3 months of lookback; horizon defaults to next trading week (5 days).
WINDOW = 60
HORIZON = 5


class _BaseScaler:
    def fit(self, arr: np.ndarray):
        raise NotImplementedError

    def transform(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_component(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """Inverse-transform a single feature component by index from a 1D array in the scaler's space.
        This is useful when y is a single feature (e.g., close) from a multi-feature scaler.
        """
        raise NotImplementedError


class MinMaxScaler(_BaseScaler):
    def __init__(self):
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        self.min_ = np.min(arr, axis=0).astype(np.float32)
        self.max_ = np.max(arr, axis=0).astype(np.float32)
        # prevent divide by zero later
        self.max_ = np.where(self.max_ == self.min_, self.min_ + 1.0, self.max_)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return (arr - self.min_) / (self.max_ - self.min_)

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return self.min_ + arr * (self.max_ - self.min_)

    def inverse_component(self, arr: np.ndarray, idx: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        # idx bounds will be handled by numpy broadcasting
        return self.min_[idx] + arr * (self.max_[idx] - self.min_[idx])


class StandardScaler(_BaseScaler):
    """Adapter around scikit-learn's StandardScaler with a NumPy fallback.

    Exposes the same small interface used in this pipeline, including
    inverse_component(idx) for single-target inversion from a multi-feature scaler.
    """

    def __init__(self):
        if _SKLEARN_AVAILABLE:
            self._sk = _SKStandardScaler()
            self.mean_: Optional[np.ndarray] = None
            self.std_: Optional[np.ndarray] = None
            self._impl = 'sklearn'
        else:
            # Fallback to a lightweight NumPy implementation
            self._sk = None
            self.mean_ = None
            self.std_ = None
            self._impl = 'numpy'

    def fit(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        arr2d = arr[:, None] if arr.ndim == 1 else arr
        if self._impl == 'sklearn' and self._sk is not None:
            self._sk.fit(arr2d)
            # Store attributes to match our previous API
            self.mean_ = self._sk.mean_.astype(np.float32)
            # sklearn uses "scale_" (std-dev)
            self.std_ = self._sk.scale_.astype(np.float32)
        else:
            self.mean_ = np.mean(arr2d, axis=0).astype(np.float32)
            std = np.std(arr2d, axis=0).astype(np.float32)
            self.std_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr2d = arr[:, None] if arr.ndim == 1 else arr
        if self._impl == 'sklearn' and self._sk is not None:
            out = self._sk.transform(arr2d)
        else:
            out = (arr2d - self.mean_) / self.std_
        return out

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr2d = arr[:, None] if arr.ndim == 1 else arr
        if self._impl == 'sklearn' and self._sk is not None:
            out = self._sk.inverse_transform(arr2d)
        else:
            out = self.mean_ + arr2d * self.std_
        return out

    def inverse_component(self, arr: np.ndarray, idx: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        # Use stored params to invert just one feature
        return self.mean_[idx] + arr * self.std_[idx]


def make_windows(series: np.ndarray, window: int = WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)
    y = np.array(y)
    return X[..., None], y

def make_windows_multi(features: np.ndarray, target: np.ndarray, window: int = WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """Create windows for multi-feature inputs with a 1D target series.
    features: shape (N, F); target: shape (N,). Returns X:(N-window, window, F), y:(N-window,).
    """
    X, y = [], []
    N = len(target)
    for i in range(N - window):
        X.append(features[i:i+window, :])
        y.append(target[i+window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_and_forecast(prices: Union[List[float], Dict[str, List[float]], List[Tuple[float, float]]], model_name: str,
                       window: int = WINDOW, horizon: int = HORIZON,
                       epochs: int = 8, lr: float = 1e-3,
                       verbose: int = 0,
                       test_split: Optional[float] = None,
                       dropout: float = 0.0,
                       batch_size: int = 32,
                       progress_callback: Optional[callable] = None) -> Dict:
    """Train a small model on close prices and produce iterative forecasts.

    Returns dict with: predictions (list), metrics (mocked minimal), window/horizon.
    """
    # Normalize input into either 1D close-only array or 2D [open, close] features with close as target
    use_multi = False
    target_idx = 0  # index of close in features when multi
    if isinstance(prices, dict) and 'close' in prices:
        close = np.asarray(prices['close'], dtype=np.float32)
        if 'open' in prices and prices['open'] is not None:
            openv = np.asarray(prices['open'], dtype=np.float32)
            L = min(len(openv), len(close))
            features_raw = np.column_stack([openv[:L], close[:L]]).astype(np.float32)
            target = features_raw[:L, 1]
            use_multi = True
            target_idx = 1
        else:
            features_raw = close.astype(np.float32)
            target = features_raw
    elif isinstance(prices, list) and len(prices) > 0 and isinstance(prices[0], (list, tuple)):
        arr = np.asarray(prices, dtype=np.float32)
        if arr.shape[1] >= 2:
            features_raw = arr[:, :2]
            target = features_raw[:, 1]
            use_multi = True
            target_idx = 1
        else:
            features_raw = arr[:, 0]
            target = features_raw
    else:
        features_raw = np.asarray(prices, dtype=np.float32)
        target = np.asarray(prices, dtype=np.float32)

    if len(target) < window + 10:
        raise ValueError("Not enough data to train the model.")

    # Select scaler: use StandardScaler for all models (LSTM and GRU)
    # This tends to stabilize training across tickers with different price scales.
    scaler = StandardScaler()
    scaler_name = 'standard'

    # Choose scaler fit range: train slice if test_split provided, else full series
    scaler_source = 'full'
    if test_split and 0.0 < test_split < 0.9:
        cut = int(len(target) * (1 - test_split))
        cut = max(cut, window + 5)
        if cut < len(target):
            fit_prices = features_raw[:cut] if use_multi else target[:cut]
            scaler_source = 'train'
        else:
            fit_prices = features_raw if use_multi else target
    else:
        fit_prices = features_raw if use_multi else target

    scaler.fit(fit_prices)
    if use_multi:
        scaled_features = scaler.transform(features_raw)
        scaled_target = scaled_features[:, target_idx]
        X, y = make_windows_multi(scaled_features, scaled_target, window)
        input_shape = (window, scaled_features.shape[1])
    else:
        scaled = scaler.transform(target)
        X, y = make_windows(scaled, window)
        input_shape = (window, 1)

    builder = get_model_builder(model_name)
    model = builder(input_shape, dropout=float(dropout))
    # Early stopping setup
    callbacks = []
    try:
        from tensorflow.keras.callbacks import Callback

        # Per-epoch progress reporter (always available)
        class _ProgressReporter(Callback):
            def __init__(self, total_epochs: int, hook):
                super().__init__()
                self.total = int(total_epochs)
                self.hook = hook
                self.last_loss = None

            def on_epoch_end(self, epoch, logs=None):
                self.last_loss = None
                if logs:
                    self.last_loss = logs.get('val_loss') or logs.get('loss')
                if callable(self.hook):
                    # epoch is zero-based; report 1-based
                    self.hook(int(epoch) + 1, self.total, None if self.last_loss is None else float(self.last_loss))

    except Exception:
        _ProgressReporter = None  # type: ignore
    # Split train/test if requested
    metrics: Dict = {}
    test_info: Optional[Dict] = None
    if test_split and 0.0 < test_split < 0.9 and len(X) > 10:
        split_idx = int(len(X) * (1 - test_split))
        split_idx = max(split_idx, 8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # Compute naive baseline (predict last value in the window) for reporting only
        if use_multi:
            naive_pred = X_test[:, -1, target_idx]
        else:
            naive_pred = X_test[:, -1, 0]
        naive_mse = float(np.mean((naive_pred - y_test) ** 2))

        # Progress reporter
        pr = None
        if progress_callback is not None and '_ProgressReporter' in locals() and _ProgressReporter is not None:
            try:
                pr = _ProgressReporter(epochs, progress_callback)  # type: ignore
                callbacks.append(pr)
            except Exception:
                pr = None

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=int(batch_size),
            verbose=verbose,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )

        # Test metrics on original scale
        test_pred = model.predict(X_test, verbose=0).squeeze()
        if use_multi:
            y_test_inv = scaler.inverse_component(y_test, target_idx)
            test_pred_inv = scaler.inverse_component(test_pred, target_idx)
        else:
            y_test_inv = scaler.inverse(y_test)
            test_pred_inv = scaler.inverse(test_pred)
        rmse = float(np.sqrt(np.mean((test_pred_inv - y_test_inv) ** 2)))
        mae = float(np.mean(np.abs(test_pred_inv - y_test_inv)))
        # MAPE on original scale; avoid division by zero by adding small epsilon
        eps = 1e-8
        mape = float(np.mean(np.abs((y_test_inv - test_pred_inv) / (np.abs(y_test_inv) + eps)))) * 100.0
        # Optional R2
        try:
            ss_res = float(np.sum((y_test_inv - test_pred_inv) ** 2))
            ss_tot = float(np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)) or 1.0
            r2 = 1.0 - ss_res / ss_tot
        except Exception:
            r2 = None

        trained_epochs = int(len(history.history.get('loss', [])) or epochs)
        stopped_early = False
        try:
            v = history.history.get('val_loss', [])
            best_val = float(np.min(v)) if len(v) > 0 else None
        except Exception:
            best_val = None

        metrics = {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'r2': None if r2 is None else round(float(r2), 4),
            'epochs': trained_epochs,
            'stopped_early': stopped_early,
            'best_val_loss': None if best_val is None else round(best_val, 6),
            'naive_val_loss': round(float(naive_mse), 6),
            'scaler_source': scaler_source,
            'scaler': scaler_name,
        }
        # This split index is in terms of windows; it aligns with fitted array length
        test_info = {
            'split_index': int(split_idx)
        }
    else:
        # Estimate validation slice (last 10%) only for metrics baselines
        val_len = max(int(len(X) * 0.1), 1)
        X_val = X[-val_len:]
        y_val = y[-val_len:]
        naive_pred = X_val[:, -1, target_idx] if use_multi else X_val[:, -1, 0]
        naive_mse = float(np.mean((naive_pred - y_val) ** 2))

        pr = None
        if progress_callback is not None and '_ProgressReporter' in locals() and _ProgressReporter is not None:
            try:
                pr = _ProgressReporter(epochs, progress_callback)  # type: ignore
                callbacks.append(pr)
            except Exception:
                pr = None

        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=int(batch_size),
            verbose=verbose,
            validation_split=0.1,
            callbacks=callbacks
        )

    # iterative forecasting
    preds_scaled = []
    if use_multi:
        last_window = scaled_features[-window:, :].copy()
        for _ in range(horizon):
            x_in = last_window[None, ...]
            yhat = float(model.predict(x_in, verbose=0).squeeze())
            preds_scaled.append(yhat)
            # naive open for next step: previous close
            next_open = float(last_window[-1, target_idx])
            next_row = last_window[-1, :].copy()
            next_row[0] = next_open  # open feature index 0
            next_row[target_idx] = yhat
            last_window = np.vstack([last_window[1:], next_row])
    else:
        last_window = scaled[-window:].tolist()
        for _ in range(horizon):
            x_in = np.array(last_window, dtype=np.float32)[None, ..., None]
            yhat = float(model.predict(x_in, verbose=0).squeeze())
            preds_scaled.append(yhat)
            last_window = last_window[1:] + [yhat]

    # invert scale
    if use_multi:
        preds = [float(x) for x in scaler.inverse_component(np.array(preds_scaled, dtype=np.float32), target_idx)]
    else:
        preds = [float(x) for x in scaler.inverse(np.array(preds_scaled, dtype=np.float32))]

    # If no test split branch was taken, compute train-slice metrics on original scale
    if not metrics:
        train_pred = model.predict(X, verbose=0).squeeze()
        if use_multi:
            y_inv = scaler.inverse_component(y, target_idx)
            train_pred_inv = scaler.inverse_component(train_pred, target_idx)
        else:
            y_inv = scaler.inverse(y)
            train_pred_inv = scaler.inverse(train_pred)
        rmse = float(np.sqrt(np.mean((train_pred_inv - y_inv) ** 2)))
        mae = float(np.mean(np.abs(train_pred_inv - y_inv)))
        eps = 1e-8
        mape = float(np.mean(np.abs((y_inv - train_pred_inv) / (np.abs(y_inv) + eps)))) * 100.0

        trained_epochs = int(len(history.history.get('loss', [])) or epochs)
        stopped_early = False
        try:
            v = history.history.get('val_loss', [])
            best_val = float(np.min(v)) if len(v) > 0 else None
        except Exception:
            best_val = None

        metrics = {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'r2': None,
            'epochs': trained_epochs,
            'stopped_early': stopped_early,
            'best_val_loss': None if best_val is None else round(best_val, 6),
            'naive_val_loss': round(float(naive_mse), 6),
            'scaler_source': scaler_source,
            'scaler': scaler_name,
        }

    # Build fitted (in-sample) predictions across the entire historical windows
    _fitted_scaled = model.predict(X, verbose=0).squeeze()
    if use_multi:
        _fitted = scaler.inverse_component(_fitted_scaled, target_idx)
    else:
        _fitted = scaler.inverse(_fitted_scaled)

    return {
        'predictions': preds,
        'metrics': metrics,
        'test_info': test_info,
        'window': window,
        'horizon': horizon,
        'dropout': float(dropout),
        'batch_size': int(batch_size),
        'scaler': scaler_name,
        'fitted': [float(x) for x in np.asarray(_fitted).reshape(-1)],
        'features': 'open_close' if use_multi else 'close',
    }
