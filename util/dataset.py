import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras

import pickle
import random
import math


class Dataset:
    def __init__(self, meta=None, filename="res_task_a.csv", winSize=144, horizon=0, resource='cpu',
                 train_split=0.8):
        # Definition of all the instance attributes
        # Name of the experiment
        self.name = filename

        # Training instances
        self.X_train = []
        # Test instances
        self.X_test = []
        # Training labels
        self.y_train = []
        # Test labels
        self.y_test = []

        self.X = []
        self.y = []

        # Features
        self.train_features = []
        self.test_features = []

        # Column to predict
        self.attribute = "avg" + resource
        self.channels = 1

        # Input files
        self.data_file = self.name  # "res_task_e.csv"  # "req_win_a.csv"  # "winDataset.csv"  # None
        self.data_path = './saved_data/'

        # Train/test split
        self.train_split = train_split

        # Type of  data normalization used
        self.normalization = "minmax" 
        self.scalers = []

        # Input window
        self.window_size = winSize  
        self.stride = 1
        self.output_window = 1
        self.start_horizon = horizon  
        self.batch_size = 32

        # Time start
        self.time_conversion_factor = 1e6
        self.time_start = time.mktime(time.strptime("01.05.2011 00:10:00", "%d.%m.%Y %H:%M:%S"))
        self.number_timestamps = 24 * 7
        self.embedding_time_size = 8

        # Number of samples
        self.samp_interval = 300  # 5 mins
        self.samp_time = 2505600  # 29 days in seconds
        self.nr_sample = int(self.samp_time / self.samp_interval)

        self.meta = meta

        self.verbose = 1

    def data_save(self, name):
        with open(self.data_path + name, 'wb') as file:
            pickle.dump(self, file)
            print("File saved in " + self.data_path + name)

    def data_load(self, name):
        with open(self.data_path + name, 'rb') as file:
            return pickle.load(file)

    def data_summary(self):
        print('Training', self.X_train.shape, 'Testing', self.X_test.shape)

    def dataset_creation(self):
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")
        df = pd.read_csv(self.data_path + self.data_file)

        if self.meta is not None:
            df['date'] = pd.to_datetime(df['time'], unit='us')
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour_of_day'] = df['date'].dt.hour

        mask = df.index < int(df.shape[0] * self.train_split)

        df_train = df[mask]
        df_test = df[~ mask]
        self.train_features = df_train["time"] 
        self.test_features = df_test["time"] 
        self.X = df[self.attribute].to_numpy() 
        self.y_train = df_train[self.attribute].to_numpy() 
        self.y_test = df_test[self.attribute].to_numpy() 
        if self.channels == 1:
            self.X = self.X.reshape(-1, 1)
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        self.X_train = df_train[self.attribute].to_numpy()
        self.X_test = df_test[self.attribute].to_numpy()

        split_value = int(self.X.shape[0] * self.train_split)
        self.X, self.y = self.windowed_dataset(self.X)

        self.X_train = self.X[:split_value]
        self.y_train = self.y[:split_value]
        self.X_test = self.X[split_value:]
        self.y_test = self.y[split_value:]

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)

        if self.verbose:
            print("Data size ", self.X.shape)

        if self.verbose:
            print("Training size ", self.X_train.shape)
            print("Training labels size", self.y_train.shape)

        if self.verbose:
            print("Test size ", self.X_test.shape)
            print("Test labels size", self.y_test.shape)

        # Normalization
        self.scalers = {}
        if self.normalization is not None:
            if self.verbose:
                print("Data normalization")
            for i in range(self.channels):
                if self.normalization == "standard":
                    self.scalers[i] = StandardScaler()
                elif self.normalization == "minmax":
                    self.scalers[i] = MinMaxScaler((-1, 1)) 
                self.X_train[:, :, i] = self.scalers[i].fit_transform(self.X_train[:, :, i])
                self.X_test[:, :, i] = self.scalers[i].transform(self.X_test[:, :, i])
                self.y_train = self.scalers[i].fit_transform(self.y_train)

                self.y_test = self.scalers[i].transform(self.y_test)

        if self.meta is not None:
            if self.verbose:
                print("Metadata")
            if self.meta == "categorical":
                tmp1 = np.zeros(
                    (self.X_train.shape[0] - self.window_size, self.X_train.shape[1] + 2, self.X_train.shape[2]))
                tmp1[:, :-2, :] = self.X_train[self.window_size:]
                tmp1[:, -2, :] = np.reshape(df_train['day_of_week'].iloc[self.window_size:].values,
                                            (self.X_train.shape[0] - self.window_size, 1))
                tmp1[:, -1, :] = np.reshape(df_train['hour_of_day'].iloc[self.window_size:].values,
                                            (self.X_train.shape[0] - self.window_size, 1))
                self.X_train = tmp1

                tmp2 = np.zeros((self.X_test.shape[0], self.X_test.shape[1] + 2, self.X_test.shape[2]))
                tmp2[:, :-2, :] = self.X_test
                tmp2[:, -2, :] = np.reshape(df_test['day_of_week'].iloc[self.window_size:].values,
                                            (self.X_test.shape[0], 1))
                tmp2[:, -1, :] = np.reshape(df_test['hour_of_day'].iloc[self.window_size:].values,
                                            (self.X_test.shape[0], 1))
                self.X_test = tmp2

    def windowed_dataset(self, series):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(self.window_size + 1, stride=self.stride, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

        inputs, targets = [], []
        a = list(dataset.as_numpy_iterator())
        for i, (X, y) in enumerate(a):
            if i == len(a) - self.start_horizon:
                break
            inputs.append(X)
            targets.append(a[i + self.start_horizon][1])
        inputs = np.array(inputs)
        targets = np.vstack(targets)
        return inputs, targets
