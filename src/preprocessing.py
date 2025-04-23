import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_data(df, seq_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = create_sequences(scaled_data, seq_length)
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:], scaler