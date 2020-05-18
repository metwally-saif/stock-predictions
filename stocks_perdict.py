import math
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def create_training_data(csv_file):
    """
    this function read the csv file and make the training data

    :param csv_file: historical stocks data in csv form
    :return: the scaled data and training x and y
    """

    df = pd.read_csv(csv_file, parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)

    data = df.filter(['Close'])
    dataset = data.values

    train_data_len = math.ceil(len(dataset) * .8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:train_data_len, :]
    x_train = []
    y_train = []

    if train_data_len < 1000:
        v = int(30)
        for i in range(v, len(train_data)):
            x_train.append(train_data[i - v:i, 0])
            y_train.append(train_data[i, 0])
    elif train_data_len < 200:
        v = int(10)
        print('The predictions will be about 30% right')
        for i in range(v, len(train_data)):
            x_train.append(train_data[i - v:i, 0])
            y_train.append(train_data[i, 0])
    else:
        v = int(math.ceil(len(train_data) * 0.1))
        for i in range(v, len(train_data)):
            x_train.append(train_data[i - v:i, 0])
            y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    print('train : ', x_train.shape)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_sequential(x_train, y_train)
    x_test, y_test = create_test_data(scaled_data, train_data_len, dataset, v)
    predict(model=model, x_test=x_test, y_test=y_test, scaler=scaler, data=data, train_data_len=train_data_len)


def create_sequential(xT, yT):
    """
    this function use (LSTM , Dense) to make the layers of the neural network

    :param xT: train x data
    :param yT: train y data
    :return: the trained model
    """

    model = Sequential()
    model.add(LSTM((math.ceil(xT.shape[1] / 2)), return_sequences=True, input_shape=(xT.shape[1], 1)))
    model.add(LSTM((math.ceil(xT.shape[1] / 2)), return_sequences=False))
    model.add(Dense((math.ceil(xT.shape[1] / 3))))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(xT, yT, batch_size=1, epochs=2)
    return model


def create_test_data(scaled_data, train_data_len, dataset, v):
    """
    :param scaled_data: this use the MinMaxScaler on the data values
    :param train_data_len: the length of the training data
    :param dataset: data values
    :param v: the calculated constant that adjust the training model
    :return: test x and y
    """

    test_data = scaled_data[train_data_len - v:, :]
    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range(v, len(test_data)):
        x_test.append(test_data[i - v:i, 0])

    x_test = np.array(x_test)

    print('Test shape: ', x_test.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test


def predict(model, x_test, y_test, scaler, data, train_data_len):
    """
    it predict the close price and calculate RNSE

    :param model:the trained model
    :param x_test: x test data
    :param y_test: y test data
    :param scaler: this use the MinMaxScaler
    :param data: filtered close prices
    :param train_data_len: the length of the train data
    :return: it returns the RNSE value and the predictions
    """

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rnse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print('RNSE= ', rnse)

    train = data[:train_data_len]
    valid = data[train_data_len:]
    valid['predictions'] = predictions

    ploting(train, valid)


def ploting(train, valid):
    """
    this function makes the graph

    :param train: the data which the model trained on
    :param valid: the predictions from the model
    :return:it returns a graph with training values , actual values and their predictions
    """

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'predictions']])
    plt.legend(['Training Value', 'Actual Value', 'Predictions'], loc='lower right')
    plt.show()


create_training_data('csv file')
