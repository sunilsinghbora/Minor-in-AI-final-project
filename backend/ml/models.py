from typing import Tuple

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input
    from tensorflow.keras.optimizers import Adam
except Exception as e:  # pragma: no cover
    tf = None
    Sequential = None
    LSTM = GRU = Dropout = Dense = Input = object


def _compile(model, lr: float = 1e-3):
    """Compile a Keras model with a robust default.

    What it does: sets optimizer and loss/metrics suited for regression.
    Called by: build_gru (and could be reused by other builders).
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not available; install tensorflow-cpu.")
    # Default compile (used by GRU): robust Huber loss + RMSE/MAE metrics
    model.compile(
        optimizer=Adam(lr),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()],
    )
    return model


def build_lstm(input_shape: Tuple[int, int], dropout: float = 0.0):
    """Create and compile the LSTM model used for forecasting.

    What it does: builds LSTM(64)x2 -> Dense(128,relu) -> Dropout -> Dense(1),
    compiles with Adam + MAE loss and RMSE metric.
    Called by: pipeline.train_and_forecast via get_model_builder('lstm').
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not available; install tensorflow-cpu.")
    d = float(dropout)
    model = Sequential([
        # First LSTM layer with sequences and explicit input shape
        LSTM(64, return_sequences=True, input_shape=input_shape),
        # Second LSTM layer without sequences
        LSTM(64, return_sequences=False),
        # Dense head + configurable dropout
        Dense(128, activation='relu'),
        Dropout(d if d > 0 else 0.0),
        Dense(1)
    ])
    # Compile as in the provided example
    model.compile(
        optimizer=Adam(1e-3),
        loss='mae',
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def build_gru(input_shape: Tuple[int, int], dropout: float = 0.0):
    """Create and compile a GRU-based regression model.

    What it does: builds GRU(64) -> Dropout -> Dense(32,relu) -> Dense(1),
    compiles with Adam + Huber loss and RMSE/MAE metrics.
    Called by: pipeline.train_and_forecast via get_model_builder('gru').
    """
    d = float(dropout)
    model = Sequential([
        Input(shape=input_shape),
        # Slightly larger GRU to help capture curvature
        GRU(64),
        Dropout(d if d > 0 else 0.0),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return _compile(model)


def get_model_builder(name: str):
    """Return a model-construction function by name.

    What it does: maps 'lstm'/'gru' to their builders; defaults to LSTM.
    Called by: pipeline.train_and_forecast to instantiate the chosen model.
    """
    name = (name or '').lower()
    return {
        'lstm': build_lstm,
        'gru': build_gru,
    }.get(name, build_lstm)