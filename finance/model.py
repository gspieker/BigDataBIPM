from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM



def train_and_predict_model(asset, data):
    # Preprocess data
    training_data_len, scaled_data, scaler = preprocess_data(data['Close'])

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

