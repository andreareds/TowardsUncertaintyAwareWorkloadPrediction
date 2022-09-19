import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from sklearn.utils import shuffle

from tensorflow import keras

import pickle
import random
import math


class Dataset:
    def __init__(self, meta=False, filenames=["res_task_a.csv"], winSize=144, horizon=0, resource='cpu',
                 bivariate=False, shuffle=False):
        # Definition of all the instance attributes
        # Name of the experiment
        self.name = filenames[0] + '-' + filenames[-1]
        print("FILES:", self.name)

        # Training instances
        self.X_train = []
        self.X_train_total = []
        # Test instances
        self.X_test = []
        self.X_test_total = []
        # Training labels
        self.y_train = []
        self.y_train_total = []
        # Test labels
        self.y_test = []
        self.y_test_total = []

        self.X = []
        self.X_total = []
        self.y = []
        self.y_total = []

        # Features
        self.train_features = []
        self.test_features = []

        # Column to predict
        if bivariate:
            self.attribute = ['avgcpu', 'avgmem']
            self.channels = 2

        else:
            self.attribute = "avg" + resource
            self.channels = 1

        # Input files
        self.data_file = filenames
        self.data_path = './saved_data/'

        # Train/test split
        self.train_split = 0.8

        # Type of  data normalization used
        self.normalization = None
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
        self.meta_day_train = []
        self.meta_day_test = []
        self.meta_hour_train = []
        self.meta_hour_test = []
        self.pos_train = []
        self.pos_test = []

        self.shuffle = shuffle

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

        print("FILES:", self.data_file)
        for f in self.data_file:
            df = pd.read_csv(self.data_path + f)

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
                print("Dataset ", f, "loaded")

            self.X_train_total.append(self.X_train)
            self.y_train_total.append(self.y_train)

            self.X_test_total.append(self.X_test)
            self.y_test_total.append(self.y_test)
            self.X_total.append(self.X)
            self.y_total.append(self.y)

            if self.meta is not None:
                self.meta_day_train.append(df_train['day_of_week'])
                self.meta_day_test.append(df_test['day_of_week'].iloc[self.window_size:])
                self.meta_hour_train.append(df_train['hour_of_day'])
                self.meta_hour_test.append(df_test['hour_of_day'].iloc[self.window_size:])
                if self.meta == 'positional':
                    max_position = self.number_timestamps
                    d_model = self.embedding_time_size
                    pos_enc = self.positional_encoding(max_position, d_model)
                    indexes = df['day_of_week'].values * 24 + df['hour_of_day'].values
                    dataset = tf.data.Dataset.from_tensor_slices(pos_enc[indexes])
                    dataset = dataset.window(self.window_size, stride=self.stride, shift=1, drop_remainder=True)
                    dataset = dataset.flat_map(lambda window: window.batch(self.window_size))
                    a = list(dataset.as_numpy_iterator())
                    inputs = []
                    for i, X in enumerate(a):
                        inputs.append(X)
                    a = np.array(inputs)
                    self.pos_train.append(a[:self.X_train.shape[0]])
                    self.pos_test.append(a[-self.X_test.shape[0]:])

        self.X = np.concatenate(self.X_total)
        self.y = np.concatenate(self.y_total)
        self.X_train = np.concatenate(self.X_train_total)
        self.X_test = np.concatenate(self.X_test_total)
        self.y_train = np.concatenate(self.y_train_total)
        self.y_test = np.concatenate(self.y_test_total)

        if self.meta is not None:
            if self.meta == 'categorical':
                self.meta_day_train = np.concatenate(self.meta_day_train)
                self.meta_day_test = np.concatenate(self.meta_day_test)
                self.meta_hour_train = np.concatenate(self.meta_hour_train)
                self.meta_hour_test = np.concatenate(self.meta_hour_test)
            elif self.meta == 'positional':
                self.pos_train = np.concatenate(self.pos_train)
                self.pos_test = np.concatenate(self.pos_test)

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
                    self.scalers[i] = MinMaxScaler()
                self.X_train[:, :, i] = self.scalers[i].fit_transform(self.X_train[:, :, i])
                self.X_test[:, :, i] = self.scalers[i].transform(self.X_test[:, :, i])
                self.y_train = self.scalers[i].fit_transform(self.y_train)
                self.y_test = self.scalers[i].transform(self.y_test)

        if self.meta is not None:
            if self.verbose:
                print("Metadata")
            if self.meta == "categorical":
                tmp1 = np.zeros((self.X_train.shape[0], self.X_train.shape[1] + 2, self.X_train.shape[2]))
                tmp1[:, :-2, :] = self.X_train
                tmp1[:, -2, :] = np.expand_dims(self.meta_day_train, axis=1)
                tmp1[:, -1, :] = np.expand_dims(self.meta_hour_train, axis=1)
                self.X_train = tmp1

                tmp2 = np.zeros((self.X_test.shape[0], self.X_test.shape[1] + 2, self.X_test.shape[2]))
                tmp2[:, :-2, :] = self.X_test[:, :, 0]
                tmp2[:, -2, :] = np.expand_dims(self.meta_day_test, axis=1)
                tmp2[:, -1, :] = np.expand_dims(self.meta_hour_test, axis=1)
                self.X_test = tmp2
            elif self.meta == 'positional':
                tmp1 = np.zeros(
                    (self.X_train.shape[0], self.X_train.shape[1],
                     self.X_train.shape[2] + self.embedding_time_size))
                tmp1[:, :, 0] = self.X_train[:, :, 0]
                tmp1[:, :, 1:] = self.pos_train
                self.X_train = tmp1
                tmp2 = np.zeros(
                    (self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2] + d_model))
                tmp2[:, :, 0] = self.X_test[:, :, 0]
                tmp2[:, :, 1:] = self.pos_test
                self.X_test = tmp2

        if self.shuffle:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=0)

    def positional_encoding(self, max_position, d_model, min_freq=1e-4):
        position = np.arange(max_position)
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_enc = position.reshape(-1, 1) * freqs.reshape(1, -1)
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

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
