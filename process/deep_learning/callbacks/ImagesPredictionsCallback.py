import numpy as np
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from .CustomCallBack import CustomCallBack

class ImagesPredictionsCallback(CustomCallBack):
    def __init__(self, x_valid, y_valid, le, file_writer):
        super().__init__(x_valid, y_valid, le, file_writer)
        
    def on_train_end(self, logs=None):
        test_pred = (self.model.predict(self.x_valid) > 0.5).astype("int32")
        test_pred = self.le.inverse_transform(np.argmax(test_pred, axis=-1))
        test_true = self.le.inverse_transform(np.argmax(self.y_valid, axis=-1))
        
        figure = plt.figure(figsize=(15, 15))
        for i in range(25):
            plt.subplot(5, 5, i + 1, title=f"predicted: {test_pred[i]}\ntrue: {test_true[i]}")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_valid[i], cmap='gray')
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        with self.file_writer.as_default():
            tf.summary.image("Training data", image, step=0)