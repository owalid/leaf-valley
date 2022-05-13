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
from callbacks.ClassificationReportCallback import ClassificationReportCallback
from callbacks.ImagesPredictionsCallback import ImagesPredictionsCallback
from metrics import recall_m, precision_m, f1_m
from sklearn import preprocessing
from sklearn.metrics import average_precision_score
sys.path.append('../../utilities')
from utils import get_dataset
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K
from models import vgg_16


def get_all_models(input_shape, num_classes):
    final_models = []
    base_models = {
        'VGG16': {
            'base': tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet'),
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        },
        'VGG19': {
            'base': tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights='imagenet'),
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        },
        'AlexNet': {
            'base': tf.keras.applications.alexnet.AlexNet(input_shape=input_shape, include_top=False, weights='imagenet'),
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        },
        'ResNet50': {
            'base': tf.keras.applications.resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet'),
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        },
        'InceptionV3': {
            'base': tf.keras.applications.inception_v3.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet'),
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        },
        'EfficientNetB0': {
            'base': tf.keras.applications.efficientnet.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet'),
            'preprocess_input': tf.keras.applications.vgg16.preprocess_input
        }
    }
    
    for base_model in base_models:
        print(f"base_model: {base_model}")
        preprocess_input = base_models[base_model]['preprocess_input']
        base_model = base_models[base_model]['base']
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
        final_models.append({'model': model, 'name': base_model})

    return final_models

def run_all_models(x_train, x_valid, y_train, y_valid, epochs=1):
    models = get_all_models()
    
    for model in models:
        name_model = model['name']
        print(f"model: {name_model}")
        current_model = model['model']
        base_dir = '../../logs/tensorboard'
        os.system(f'rm -rf {base_dir}/*')
        logdir = f"{base_dir}/{name_model}/{datetime.now().strftime('%Y%m%d')}"
        
        # Define the basic TensorBoard callback.
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True, histogram_freq=1)
        
        ''' Callbacks '''
        file_writer = tf.summary.create_file_writer(logdir + '/cm')
        callbacks = [
            tensorboard_cb,
            ConfusionMatrixCallback(x_valid, y_valid, le, file_writer),
            ImagesPredictionsCallback(x_valid, y_valid, le, file_writer)
        ]
        
        base_learning_rate = 0.001
        current_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics=[
                        'accuracy',
                        'categorical_accuracy',
                        'sparse_categorical_accuracy',
                        'sparse_top_k_categorical_accuracy',
                        'mean_squared_error',
                        'mean_absolute_error',
                        'mean_absolute_percentage_error',
                        'mean_squared_logarithmic_error',
                        # CUSTOM METRICS FILE: metrics.py
                        f1_m,
                        precision_m,
                        recall_m,
                        tf.keras.metrics.AUC()
                    ])
        current_model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, callbacks=callbacks)
        
        
    
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
    classes = np.array(hf['classes']).astype(str)
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
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True, histogram_freq=1)
    
    ''' Callbacks '''
    file_writer = tf.summary.create_file_writer(logdir + '/cm')
    callbacks = [
        tensorboard_cb,
        ConfusionMatrixCallback(x_valid, y_valid, le, file_writer),
        ClassificationReportCallback(x_valid, y_valid, le, file_writer),
        ImagesPredictionsCallback(x_valid, y_valid, le, file_writer)
    ]
    
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[
                    'accuracy',
                    'mean_absolute_error',
                    'categorical_accuracy',
                    'categorical_crossentropy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.FalsePositives(),
                    tf.keras.metrics.FalseNegatives(),
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives(),
                    
                    # CUSTOM METRICS FILE: metrics.py
                    f1_m,
                    precision_m,
                    recall_m
                ])
    initial_epochs = 1
    
    # %tensorboard --logdir ../logs/tensorbord/
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=initial_epochs, callbacks=callbacks)

    print("run tensorboard")
    os.system(f'tensorboard --logdir={logdir}')
    
    print(f"Model trained for {initial_epochs} epochs")
    print(f"Model summary: {model.summary()}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
