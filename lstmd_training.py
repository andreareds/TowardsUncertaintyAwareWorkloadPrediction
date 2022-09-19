import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from models import LSTMD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from util import dataset, plot_training, save_results, multipleDataset
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
wins = [288]
hs = [2]
resources = ['cpu', 'mem']
clusters = ['gc19_a', 'gc19_b', 'gc19_c', 'gc19_d', 'gc19_e', 'gc19_f', 'gc19_g', 'gc19_h', 'gc11', 'ali18', 'ali20_c',
            'ali20_g']

files = []
[files.append('preprocessed/' + c + '.csv') for c in clusters]

bivariate = True
model = 'LSTMD'
type = 'multibivariate' #bivariate #multiunivariate #univariate
ITERATIONS = 10

suffix = "MULTIBI" # "BI", "MULTI", ""

train_splits = [0.8, 0.6, 0.4, 0.2]
tuning_rates = [6] #, 12, 18, 24]
tuning = False

for tuning_rate in tuning_rates:
    for ts in train_splits:
        for win in wins:
            for res in resources:
                for c in clusters:
                    for h in hs:
                        mses, maes = [], []

                        if tuning:
                            experiment_name = '-' + res + '-' + '-w' + str(win) + '-h' + str(h) + \
                                              '-tuning' + str(int(tuning_rate))
                        else:
                            experiment_name = suffix + model + res + '-w' + str(win) + '-h' + str(h) + \
                                              '-ts' + str(int(ts * 100))

                        # Data creation and load
                        if type == 'univariate' or type == 'bivariate':
                            ds = dataset.Dataset(meta=False, filename='preprocessed/' + c + '.csv', winSize=win, horizon=h,
                                                resource=res, bivariate=bivariate)
                        else:
                            ds = multipleDataset.Dataset(meta=None, filenames=files, winSize=288, horizon=2,
                                                     resource=res, bivariate=bivariate, shuffle=True)

                        print(ds.name)
                        ds.dataset_creation()

                        ds.data_summary()

                        best_model, best_history, best_prediction_mean, best_prediction_std = None, None, None, None
                        best_mse = 100000

                        parameters = pd.read_csv("hyperparams/"+ type + "/" + model + "-w288-h2.csv").iloc[0]

                        dense_act = 'relu'
                        if 'relu' in parameters['first_dense_activation']:
                            dense_act = 'relu'
                        elif 'tanh' in parameters['first_dense_activation']:
                            dense_act = 'tanh'

                        p = {
                            'first_conv_dim': int(parameters['first_conv_dim']),
                            'first_conv_kernel': int(parameters['first_conv_kernel']),
                            'first_conv_activation': parameters['first_conv_activation'],
                            'cnn_layers': int(parameters['cnn_layers']),
                            'second_lstm_dim': int(parameters['second_lstm_dim']),
                            'first_dense_dim': int(parameters['first_dense_dim']),
                            'first_dense_activation': dense_act,
                            'mlp_units': np.array(parameters['mlp_units'][1:-1].split(','), dtype=np.int),
                            'dense_kernel_init': parameters['dense_kernel_init'],
                            'batch_size': int(parameters['batch_size']),
                            'epochs': int(parameters['epochs']),
                            'patience': int(parameters['patience']),
                            'optimizer':  parameters['optimizer'],
                            'lr': float(parameters['lr']),
                            'momentum': float(parameters['momentum']),
                            'decay': float(parameters['decay'])
                        }

                        print(p)

                        training_times, inference_times, tuning_times = [], [], []

                        for it in range(ITERATIONS):
                            print("ITERATION:", it, "HORIZON:", h, "WIN:", win)
                            model = LSTMD.LSTMDPredictor()
                            model.name = experiment_name

                            start = datetime.now()

                            train_model, history, prediction_mean, prediction_std, training_time, inference_time = \
                                model.training(ds.X_train, ds.y_train,
                                               ds.X_test,
                                               ds.y_test, p)
                            training_times.append(training_time)
                            inference_times.append(inference_time)

                            if tuning:
                                for i in range(int(ds.X_test.shape[0] / tuning_rate) - 1):
                                    train_model, history, tuning_time = model.tuning(
                                        ds.X_test[i * tuning_rate:(i + 1) * tuning_rate],
                                        ds.y_test[i * tuning_rate:(i + 1) * tuning_rate], p)
                                    tuning_times.append(tuning_time)

                            train_distribution = train_model(ds.X_train)
                            train_mean = train_distribution.mean().numpy()
                            train_std = train_distribution.stddev().numpy()

                            save_results.save_uncertainty_csv(train_mean, train_std,
                                                              ds.y_train,
                                                              'avg' + res,
                                                              'train-' + model.name + "-run-" + str(it), bivariate=bivariate)
                            mse = mean_squared_error(ds.y_test, prediction_mean)
                            mae = mean_absolute_error(ds.y_test, prediction_mean)
                            mses.append(mse)
                            maes.append(mae)

                            save_results.save_uncertainty_csv(prediction_mean, prediction_std,
                                                              ds.y_test[:len(prediction_mean)], 'avg' + res,
                                                              model.name + "-run-" + str(it),
                                                              bivariate=bivariate)

                            if mse < best_mse:
                                best_mse = mse
                                best_prediction_mean = prediction_mean
                                best_prediction_std = prediction_std
                                best_model = train_model
                                best_history = history

                            if tuning:
                                df_tuning_times = pd.DataFrame({'time': tuning_times})
                                df_tuning_times.to_csv("time/" + experiment_name + 'tuning_time.csv')
                            else:
                                df_training_time = pd.DataFrame({'time': training_times})
                                df_training_time.to_csv("time/" + experiment_name + 'training_time.csv')

                                df_inference_time = pd.DataFrame({'time': inference_times})
                                df_inference_time.to_csv("time/" + experiment_name + 'inference_time.csv')

                            prediction_mean = best_prediction_mean
                            prediction_std = best_prediction_std
                            history = best_history
                            train_model = best_model
                            save_results.save_errors(mses, maes, model.name)
