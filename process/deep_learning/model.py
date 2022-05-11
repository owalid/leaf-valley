import os
import io
import pandas as pd
import seaborn as sns
from datetime import datetime
import argparse as ap
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from argparse import RawTextHelpFormatter
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from callbacks.ConfusionMatrixCallback import ConfusionMatrixCallback
from callbacks.ImagesPredictionsCallback import ImagesPredictionsCallback
from sklearn import preprocessing
sys.path.append('../../utilities')
from utils import get_dataset
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def vgg_16(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the VGG16 model
        Arguments:
            image_shape -- Image width and height
        Returns:
            tf.keras.model
    '''
    
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False # freeze the base model by making it non trainable
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dense(512, activation='relu')(x)
    x = tfl.Dropout(0.5)(x)
    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model
    

if __name__ == '__main__':
    parser = ap.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-path", "--path-dataset", required=False, type=str, default='../../data/deep_learning/export/data_all_20_gray.h5', help='Path of your dataset (h5 file)')
    args = parser.parse_args()
    path_dataset = args.path_dataset
    if path_dataset.split('.')[-1] != 'h5':
        print("Please provide a h5 file")
        exit(0)
    if not os.path.exists(path_dataset):
        print("File does not exist")
        exit(0)
    hf = get_dataset(path_dataset)
    print(f"Dataset keys: {hf.keys()}")
    print(f"Dataset rgb shape: {hf['rgb_img'].shape}")
    
    
    
    img_size = (hf['rgb_img'].shape[1], hf['rgb_img'].shape[2], hf['rgb_img'].shape[3])
    # print(f"img_size: {img_size}")
    classes = np.array(hf['classes']).astype(np.str)
    print(classes[0: 5])
    labels = np.array(hf['labels'])
    # print("classes", classes)
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    class_names = le.classes_
    encoded_y = le.transform(classes)
    images = np.array(hf['rgb_img'])
    encoded_y = tf.keras.utils.to_categorical(encoded_y, num_classes=len(class_names))
    print(encoded_y[0])
    x_train, x_valid, y_train, y_valid = train_test_split(images, encoded_y, test_size=0.4, shuffle=True)
    print(x_train.shape, y_train.shape)
    # print(le.inverse_transform(encoded_y))
    model = vgg_16(img_size, len(class_names))
    
    # Analyze the results with tensorboard
    base_dir = '../../logs/tensorboard'
    os.system(f'rm -rf {base_dir}/*')
    logdir = base_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Define the basic TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True, update_freq='epoch')
    
    # TODO F1 score, precision recall, validation
    ''' Callbacks '''
    file_writer = tf.summary.create_file_writer(logdir + '/cm')
    cm_cb = ConfusionMatrixCallback(x_valid, y_valid, le, file_writer)
    images_predictions_cb = ImagesPredictionsCallback(x_valid, y_valid, le, file_writer)
    
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[
                    metrics.mean_squared_error, 
                    metrics.mean_absolute_error, 
                    metrics.mean_absolute_percentage_error,
                    metrics.categorical_accuracy
                ])
    initial_epochs = 1
    
    # %tensorboard --logdir ../logs/tensorbord/
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=initial_epochs, callbacks=[tensorboard_cb, cm_cb, images_predictions_cb])

    print("run tensorboard")
    os.system(f'tensorboard --logdir={logdir}')
    
    print(f"Model trained for {initial_epochs} epochs")
    print(f"Model summary: {model.summary()}")