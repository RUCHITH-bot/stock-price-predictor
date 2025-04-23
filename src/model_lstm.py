from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import numpy as np

def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_sequence, n_days, scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_days):
        prediction = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        future_predictions.append(prediction[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
