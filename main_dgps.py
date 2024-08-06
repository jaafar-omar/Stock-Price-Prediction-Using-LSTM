import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Define a function to load the dataset from a CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    # Assuming the CSV file has 'Date' column, you can set it as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Example usage:
    data = load_data('data.csv') 
    print(data.head())
    return data

def split_train_test(data, train_ratio=0.7):
    """
    Splits the given DataFrame into training and testing sets based on the provided ratio.

    Parameters:
        data (DataFrame): The input DataFrame.
        train_ratio (float): The ratio of data to be used for training. Default is 0.7.

    Returns:
        train (DataFrame): The training set.
        test (DataFrame): The testing set.
    """
    train_size = int(len(data) * train_ratio)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test

def min_max_scale(data, feature_range=(0, 1), columns=None):
    """
    Performs Min-Max scaling on the specified columns of the DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.
        feature_range (tuple): The range of transformed data after scaling. Default is (0, 1).
        columns (list): A list of column names to be scaled. If None, all numeric columns will be scaled.

    Returns:
        scaled_data (DataFrame): The DataFrame with scaled values.
        scaler (MinMaxScaler): The scaler object used for scaling.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns.tolist()
    scaled_data = data.copy()
    scaled_data[columns] = scaler.fit_transform(data[columns])
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    """
    Creates input sequences and corresponding target values for training.

    Parameters:
        data (numpy.ndarray): The input data.
        sequence_length (int): The length of input sequences.

    Returns:
        x_train (numpy.ndarray): Input sequences.
        y_train (numpy.ndarray): Corresponding target values.
    """
    x_train = []
    y_train = []

    for i in range(sequence_length, data.shape[0]):
        x_train.append(data[i - sequence_length: i])
        y_train.append(data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('../data/data.csv')
    scaler = MinMaxScaler(feature_range=(0,1))
    train_close = data.iloc[:, 4:5].values
    data_training_array = scaler.fit_transform(train_close)
    
    # Create input sequences and target values
    sequence_length = 100
    x_train, y_train = create_sequences(data_training_array, sequence_length)
    
    # Define the model architecture
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    # Train the model
    model.fit(x_train, y_train, epochs=10)