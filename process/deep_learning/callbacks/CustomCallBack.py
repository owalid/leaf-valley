from abc import abstractclassmethod
import numpy as np
from tensorflow import keras


class CustomCallBack(keras.callbacks.Callback):
    def __init__(self, x_valid, y_valid, le, file_writer):
       super().__init__()
       self.x_valid = x_valid
       self.y_valid = y_valid
       self.le = le
       self.file_writer = file_writer

    @abstractclassmethod   
    def on_train_begin(self, logs=None):
        pass

    @abstractclassmethod
    def on_train_end(self, logs=None):
        pass

    @abstractclassmethod
    def on_epoch_begin(self, epoch, logs=None):
        pass

    @abstractclassmethod
    def on_epoch_end(self, epoch, logs=None):
        pass

    @abstractclassmethod
    def on_test_begin(self, logs=None):
        pass

    @abstractclassmethod
    def on_test_end(self, logs=None):
        pass

    @abstractclassmethod
    def on_predict_begin(self, logs=None):
        pass

    @abstractclassmethod
    def on_predict_end(self, logs=None):
        pass

    @abstractclassmethod
    def on_train_batch_begin(self, batch, logs=None):
        pass

    @abstractclassmethod
    def on_train_batch_end(self, batch, logs=None):
        pass

    @abstractclassmethod
    def on_test_batch_begin(self, batch, logs=None):
        pass

    @abstractclassmethod
    def on_test_batch_end(self, batch, logs=None):
        pass

    @abstractclassmethod
    def on_predict_batch_begin(self, batch, logs=None):
        pass

    @abstractclassmethod
    def on_predict_batch_end(self, batch, logs=None):
        pass