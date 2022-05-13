import numpy as np
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from .CustomCallBack import CustomCallBack

class ImagesPredictionsCallback(CustomCallBack):
    def __init__(self, x_valid, y_valid, le, file_writer):
        super().__init__(x_valid, y_valid, le, file_writer)
    
    def chunks(self, arr, chunk_size):
        '''
            Split array into chunks
            Args:
            arr: array to split
            chunk_size: size of chunks
            Returns:
            list of chunks
        '''
        return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]
    
    def canvas2rgb_array(self, canvas):
        """Adapted from: https://stackoverflow.com/a/21940031/959926"""
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = canvas.get_width_height()
        scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
        return buf.reshape(scale * nrows, scale * ncols, 3)
    
    def on_train_end(self, logs=None):
        test_pred = (self.model.predict(self.x_valid) > 0.5).astype("int32")
        test_pred = self.le.inverse_transform(np.argmax(test_pred, axis=-1))
        test_true = self.le.inverse_transform(np.argmax(self.y_valid, axis=-1))
        
        index = 0
        chunks_img = self.chunks(test_pred, 6)
        figures_array = []
        for chunk_img in chunks_img:
            figure = plt.figure(figsize=(7, 7))
            for i in range(len(chunk_img)):
                plt.subplot(3, 2, i + 1, title=f"predicted: {test_pred[index]}\ntrue: {test_true[index]}")
                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(self.x_valid[index], cmap='gray')
                index += 1
            figures_array.append(self.canvas2rgb_array(figure.canvas))
        with self.file_writer.as_default():
            figures_array = np.array(figures_array)
            tf.summary.image("Training data", figures_array, step=0)