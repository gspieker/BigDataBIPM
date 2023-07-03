from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1,1))

    return len(data), scaled_data, scaler

def train_model(training_data_len, scaled_data):
    # Create the training dataset
    train_data = scaled_data[0:training_data_len , :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    return model

def train_and_predict_model(asset, data):
    # Preprocess data
    training_data_len, scaled_data, scaler = preprocess_data(data)

    # Train model
    model = train_model(training_data_len, scaled_data)

    # Use model to predict future prices and scale back to original data range
    test_data = scaled_data[training_data_len - 60: , :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Prepare dataframe for plotting
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    return valid
