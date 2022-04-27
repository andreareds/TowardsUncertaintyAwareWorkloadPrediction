import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class CustomSaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.monitor = 'val_loss'
        self.dnn = model

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if logs['val_loss'] < self.dnn.best_val_loss:
            print('New best validation loss: ', logs['val_loss'])
            self.dnn.best_val_loss = logs['val_loss']
            self.dnn.model = self.dnn.train_model
            self.dnn.save_model()
            print('Model save id ', str(self.dnn.count_save - 1).zfill(4))
