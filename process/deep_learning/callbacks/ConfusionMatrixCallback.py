import numpy as np
import pandas as pd
import seaborn as sns
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from .CustomCallBack import CustomCallBack

class ConfusionMatrixCallback(CustomCallBack):
    def __init__(self, x_valid, y_valid, le, file_writer):
        super().__init__(x_valid, y_valid, le, file_writer)
        
    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        test_pred = self.model.predict(self.x_valid)
        test_pred = self.le.inverse_transform(np.argmax(test_pred, axis=-1))
        test_true = self.le.inverse_transform(np.argmax(self.y_valid, axis=-1))
        cm = confusion_matrix(test_true, test_pred)
        print(self.le.classes_)
        con_mat_df = pd.DataFrame(cm, index=self.le.classes_, columns=self.le.classes_)
        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)

        