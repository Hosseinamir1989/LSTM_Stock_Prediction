from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(window_size, n_features, learning_rate=0.0001):
    """
    Single-layer LSTM model for next-day stock price prediction.

    Architecture:
        Input (window_size × n_features)
        → LSTM(100 units)
        → Dropout(0.2)
        → Dense(1)

    Rationale:
    - Single LSTM layer to avoid unnecessary model complexity.
    - 100 hidden units to capture temporal dependencies.
    - Dropout for regularization and overfitting control.
    - Adam optimizer with a lower learning rate (0.0001) for stable convergence.
    """

    model = Sequential([
        LSTM(
            units=100,
            activation="tanh",               # cell state activation
            recurrent_activation="sigmoid",  # gate activation (standard LSTM)
            input_shape=(window_size, n_features)
        ),
        Dropout(0.2),  # regularization
        Dense(1, activation="linear")  # regression output
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )

    return model
