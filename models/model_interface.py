from tensorflow.keras.models import load_model
from tensorflow import keras
from util import custom_keras
import numpy as np


class ModelInterface:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.train_model = None
        self.verbose = False
        self.model_path = './saved_models/'
        self.count_save = 0
        self.best_val_loss = np.Inf
        self.input_shape = None

    def predict(self, X):
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        return self.model.predict(X)

    def save_model(self):
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf', save_format="tf")
        self.count_save += 1

    def training(self, X_train, y_train, X_test, y_test, p):
        pass
