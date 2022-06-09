import numpy as np
import pandas as pd
import seaborn as sns
import io
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from .CustomCallBack import CustomCallBack

class ClassificationReportCallback(CustomCallBack):
    def __init__(self, x_valid, y_valid, le, file_writer):
        super().__init__(x_valid, y_valid, le, file_writer)
        
    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        test_pred = self.model.predict(self.x_valid)
        test_pred = self.le.inverse_transform(np.argmax(test_pred, axis=-1))
        test_true = self.le.inverse_transform(np.argmax(self.y_valid, axis=-1))
        clf_report = classification_report(test_true, test_pred, target_names=self.le.classes_, output_dict=True)
        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.tight_layout()        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Classification report", image, step=epoch)

        