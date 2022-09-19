import pandas as pd
import numpy as np


def save_output_csv(preds, labels, feature, filename, bivariate=False):
    PATH = "res/output_" + filename + ".csv"
    if bivariate:
        dct = {'avgcpu': preds[:, 0],
               'labelsavgcpu': labels[:, 0],
               'avgmem': preds[:, 1],
               'labelsavgmem': labels[:, 1]
               }
    else:
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        dct = {feature: preds,
               'labels': labels}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_uncertainty_csv(preds, std, labels, feature, filename, bivariate=False):
    PATH = "res/output_" + filename + ".csv"

    if bivariate:
        std = np.array(std)
        preds = np.array(preds)
        dct = {'avgcpu': preds[:, 0],
               'stdavgcpu': std[:, 0],
               'labelsavgcpu': labels[:, 0],
               'avgmem': preds[:, 1],
               'stdavgmem': std[:, 1],
               'labelsavgmem': labels[:, 1]
               }
    else:
        preds = np.concatenate(preds, axis=0)
        std = np.concatenate(std, axis=0)
        labels = np.concatenate(labels, axis=0)
        dct = {feature: preds,
               'std': std,
               'labels': labels}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_params_csv(p, filename):
    PATH = "param/p_" + filename + ".csv"
    df = pd.DataFrame(p, index=[0])
    df.to_csv(PATH)


def save_bayes_csv(preds, min, max, labels, feature, filename):
    PATH = "res/vidp_" + filename + ".csv"
    dct = {feature: preds,
           'min': min,
           'max': max,
           'labels': labels}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_errors(mses, maes, filename):
    PATH = "res/errors_" + filename + ".csv"
    dct = {'MSE': mses,
           'MAE': maes}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)
