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
    """LSTM architecture mirroring the reference: two stacked LSTM(64),
    Dense(128, relu), Dropout, Dense(1), compiled with MAE loss and RMSE metric.
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
    name = (name or '').lower()
    return {
        'lstm': build_lstm,
        'gru': build_gru,
    }.get(name, build_lstm)