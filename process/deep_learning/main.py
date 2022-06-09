import os
from datetime import datetime
import argparse as ap
from argparse import RawTextHelpFormatter
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from sklearn import preprocessing
from custom_models import classic_cnn, alexnet, lab_process, hsv_process
from metrics import recall_m, precision_m, f1_m
import h5py
import json

# Tensorflow
import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras.models import save_model


# Callbacks
from callbacks.ConfusionMatrixCallback import ConfusionMatrixCallback
from callbacks.ClassificationReportCallback import ClassificationReportCallback
from callbacks.ImagesPredictionsCallback import ImagesPredictionsCallback


# Import from base directory
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
current_dir = current_dir[:current_dir.rfind(path.sep)]
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from utilities.utils import get_dataset


VERBOSE = False

base_models = {
    'VGG16': {
        'base': tf.keras.applications.VGG16,
        'preprocess_input': tf.keras.applications.vgg16.preprocess_input
    },
    'VGG19': {
        'base': tf.keras.applications.VGG19,
        'preprocess_input': tf.keras.applications.vgg19.preprocess_input
    },
    'RESNET50': {
        'base': tf.keras.applications.ResNet50,
        'preprocess_input': tf.keras.applications.resnet50.preprocess_input
    },
    'RESNET50V2': {
        'base': tf.keras.applications.ResNet50V2,
        'preprocess_input': tf.keras.applications.resnet_v2.preprocess_input
    },
    'INCEPTIONRESNETV2': {
        'base': tf.keras.applications.InceptionResNetV2,
        'preprocess_input': tf.keras.applications.inception_resnet_v2.preprocess_input 
    },
    'INCEPTIONV3': {
        'base': tf.keras.applications.InceptionV3,
        'preprocess_input': tf.keras.applications.inception_v3.preprocess_input
    },
    'EFFICIENTNETB0': {
        'base': tf.keras.applications.EfficientNetB0,
        'preprocess_input': tf.keras.applications.efficientnet.preprocess_input
    },
    'EFFICIENTNETB7': {
        'base': tf.keras.applications.EfficientNetB7,
        'preprocess_input': tf.keras.applications.efficientnet.preprocess_input
    },
    'XCEPTION': {
        'base': tf.keras.applications.Xception,
        'preprocess_input': tf.keras.applications.xception.preprocess_input
    },
    'CLASSIC_CNN': {
        'base': classic_cnn,
        'preprocess_input': None
    },
    'ALEXNET': {
        'base': alexnet,
        'preprocess_input': None
    },
    'LAB_PROCESS': {
        'base': lab_process,
        'preprocess_input': None
    },
    'HSV_PROCESS': {
        'base': hsv_process,
        'preprocess_input': None
    },
}


def local_print(msg):
    if VERBOSE:
        print(msg)

def get_model(input_shape, num_classes, model_name, should_train):
    base_model = base_models[model_name]
    preprocess_input = base_model['preprocess_input']

    if should_train:
        base_model = base_model['base'](input_shape=input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False # freeze the base model by making it non trainable
    else:
        base_model = base_model['base'](input_shape=input_shape, include_top=False)
        base_model.trainable = True

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

def run_models(x_train, x_valid, y_train, y_valid, model_names, input_shape, num_classes, batch_size, le, dest_logs, epochs, dest_models, should_save_model):
    keras_verbose = 1 if VERBOSE else 0
    for model_name in model_names:
        if base_models[model_name]['preprocess_input'] is None: # if is not pretrained model
            current_model = base_models[model_name]['base'](input_shape, num_classes)
        else:
            should_train = False if model_name.endswith('_PRETRAINED') else True
            current_model = get_model(input_shape, num_classes, model_name, should_train)  # get pretrained model
        local_print("==========================================================")
        local_print(f"[+] Current model: {model_name}")
        
        callbacks = get_tensorboard_callbacks(model_name, x_valid, y_valid, le, dest_logs)
        
        base_learning_rate = 0.001
        local_print(f"[+] Compile model {model_name}...")
        current_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
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
        local_print(f"[+] Fitting model {model_name}...")
        current_model.fit(x_train, y_train, verbose=keras_verbose, validation_data=(x_valid, y_valid), epochs=epochs, callbacks=callbacks, batch_size=batch_size)
        if should_save_model:
            save_model_ext(current_model, f"{dest_models}/{model_name}.h5", le=le)
            
        local_print(f"[+] Model trained for {epochs} epochs")
        local_print(f"[+] Model summary: {current_model.summary()}")
        local_print(f"[+] End training model {model_name}")
        local_print("==========================================================\n\n")

def get_tensorboard_callbacks(model_name, x_valid, y_valid, le, dest_logs):
    base_dir = dest_logs
    os.system(f'rm -rf {base_dir}/{model_name}')
    logdir = f"{base_dir}/{model_name}/{datetime.now().strftime('%Y%m%d')}"
    
    # Define the basic TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True, histogram_freq=1)
    
    ''' Callbacks '''
    file_writer = tf.summary.create_file_writer(logdir + '/cm')
    callbacks = [
        tensorboard_cb,
        ConfusionMatrixCallback(x_valid, y_valid, le, file_writer)
        # ClassificationReportCallback(x_valid, y_valid, le, file_writer)
        # ImagesPredictionsCallback(x_valid, y_valid, le, file_writer)
    ]
    return callbacks


def save_model_ext(model, filepath, overwrite=True, le=None):
    # https://stackoverflow.com/questions/44310448/attaching-class-labels-to-a-keras-model
    save_model(model, filepath, overwrite)
    if le is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['class_names'] = json.dumps(list(le.classes_))
        f.close()
        
def extract_features(hf):
    '''
        Extract X and y from dataset h5 file
    '''
    local_print(f"[+] Extract features from h5 file")
    y = np.array(hf['classes']).astype(str)
    local_print(f"[+] y created.")
    x = None
    
    for key in hf.keys():
        if key != 'classes':
            local_print(f"[+] Add {key} feature in X.")
            if len(np.array(hf[key]).shape) != 4:
                print(f"[-] Feature {key} have not 4 dim")
                exit(6)
            if x is None:
                x = np.array(hf[key])
            else:
                x = np.concatenate((x, np.array(hf[key])), axis=1)
                
    if type(x) is not np.ndarray:
        x = np.array([])
        
    return x, y
    
    
def encode_labels(y):
    '''
        Encode labels to integers
    '''
    classes = y
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    encoded_y = le.transform(classes)
    class_names = le.classes_
    encoded_y = tf.keras.utils.to_categorical(encoded_y, num_classes=len(class_names))
    return encoded_y, le, class_names
    
if __name__ == '__main__':
    models_availaibles = list(base_models.keys())
    parser = ap.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-p", "--path-dataset", required=False, type=str, default='data/deep_learning/export/data_all_20_gray.h5', help='Path of your dataset (h5 file)')
    parser.add_argument("-lt", "--launch-tensorboard", required=False, action='store_true', default=False, help='Launch tensorboard after fitting')
    parser.add_argument("-b", "--batch-size", required=False, type=int, default=32, help='Batch size')
    parser.add_argument("-e", "--epochs", required=False, type=int, default=50, help='Epoch')
    parser.add_argument("-m", "--models", required=False, type=str, help=f'Select model(s), if grid search is enabled, you can select multiple models separate by ",". example -m=vgg19,resnet50. By default is select all models.\nModels availables:\n{", ".join(models_availaibles)}.')
    parser.add_argument("-s", "--save-model", required=False, action='store_true', default=False, help='Save model')
    parser.add_argument("-dst-l", "--dest-logs", required=False, type=str, help='Destination for tensorboard logs. (default logs/tensorboard)')
    parser.add_argument("-dst-m", "--dest-models", required=False, type=str, help='Destination for model if save model is enabled')
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help='Verbose')
    args = parser.parse_args()
    print(args)
    path_dataset = args.path_dataset
    launch_tensorboard = args.launch_tensorboard
    batch_size = args.batch_size
    epochs = args.epochs
    model_names = args.models
    should_save_model = args.save_model
    dest_models = args.dest_models
    dest_logs = args.dest_logs
    VERBOSE = args.verbose
    
    
    if not os.path.exists(path_dataset):
        print("[-] File does not exist")
        exit(1)

    if path_dataset.split('.')[-1] != 'h5':
        print("[-] Please provide a h5 file")
        exit(2)
        
    
    if not dest_logs:
        dest_logs = 'logs/tensorboard'
    
    if not dest_models:
        dest_models = 'models'
        
    if not os.path.exists(dest_logs): # Create a dest_logs if not exist. 
        os.makedirs(dest_logs)
        
    if not os.path.exists(dest_models): # Create a dest_models if not exist. 
        os.makedirs(dest_models)
        
    model_names = model_names.replace(' ', '').split(',') if model_names != '' else models_availaibles
    model_names = [model_name.upper() for model_name in model_names]
    
    # Check if models correspond to base_models and not duplicated
    model_names = list(dict.fromkeys(model_names)) # delete duplicated
    model_names = [model_name for model_name in model_names if model_name in models_availaibles] # delete not in base_models

    if len(model_names) == 0:
        local_print(f"[-] No model selected, select one of the following:\n{', '.join(models_availaibles)}")
        exit(3)

    local_print(f"[+] MODELS: {','.join(model_names)}")
    
    hf = get_dataset(path_dataset)
    local_print(f"[+] Dataset keys: {hf.keys()}")
    
    if 'classes' not in hf.keys():
        print('[-] Dataset does not contain classes')
        exit(4)
    
    X, y = extract_features(hf)
    local_print(f"[+] X shape: {X.shape}")
    local_print(f"[+] y shape: {y.shape}")
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("[-] Dataset is empty, please check your dataset")
        exit(5)

    if len(X.shape) != 4:
        print("[-] Dataset is not in 4D")
        exit(6)
    
    input_shape = tuple(X.shape[1:])
    local_print(f"[+] Input shape: {input_shape}")
    y, le, class_names = encode_labels(y)
    num_classes = len(class_names)
    local_print("[+] Classes:\n\t- " + '\n\t- '.join(class_names) + "\n")
    local_print(f"[+] Numbers of classes: {num_classes}")
    
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, shuffle=True)
    
    local_print(f"[+] X_train.shape: {x_train.shape} | y_train.shape: {y_train.shape}")
    local_print(f"[+] X_valid.shape: {x_valid.shape} | y_valid.shape: {y_valid.shape}")
    
    # Run all models according to model_names arg
    run_models(x_train, x_valid, y_train, y_valid, model_names, input_shape, num_classes, batch_size, le, dest_logs, epochs, dest_models, should_save_model)
 
    if launch_tensorboard:
        print("[+] Run tensorboard")
        os.system(f'tensorboard --logdir={dest_logs}')
    