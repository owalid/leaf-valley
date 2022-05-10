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
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
sys.path.append('../../utilities')
from utils import get_dataset
import tensorflow as tf
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
    
    # def log_confusion_matrix(epoch, logs):
    #     # Use the model to predict the values from the validation dataset.
    #     test_pred_raw = model.predict(x_valid)
    #     test_pred = np.argmax(test_pred_raw, axis=1)

    #     # Calculate the confusion matrix.
    #     cm = confusion_matrix(y_valid, test_pred)
    #     # Log the confusion matrix as an image summary.
    #     figure = plot_confusion_matrix(cm, class_names=class_names)
    #     cm_image = plot_to_image(figure)

    #     # Log the confusion matrix as an image summary.
    #     with file_writer_cm.as_default():
    #         tf.summary.image("Confusion Matrix", cm_image, step=epoch)
            
    # Analyze the results with tensorboard
    base_dir = '../../logs/tensorboard'
    os.system(f'rm -rf {base_dir}/*')
    logdir = base_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Define the basic TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True, update_freq='epoch')
    
    
    ''' confusion matrix summaries '''
    file_writer = tf.summary.create_file_writer(logdir + '/cm')

    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred = (model.predict(x_valid) > 0.5).astype("int32")
        test_pred = le.inverse_transform(np.argmax(test_pred, axis=-1))
        test_true = le.inverse_transform(np.argmax(y_valid, axis=-1))
        print(test_pred.shape, test_true.shape)
        print(test_pred[0:10])
        print(test_true[0:10])
        cm = confusion_matrix(test_true, test_pred)
        print(le.classes_)
        print(le.classes_.shape)
        print(cm.shape)
        con_mat_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        print(con_mat_df)
        figure = plt.figure(figsize=(8, 8))
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
        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)

    cm_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    
    def log_images_predictions(logs):
        test_pred = (model.predict(x_valid) > 0.5).astype("int32")
        test_pred = le.inverse_transform(np.argmax(test_pred, axis=-1))
        test_true = le.inverse_transform(np.argmax(y_valid, axis=-1))
        
        figure = plt.figure(figsize=(15, 15))
        for i in range(25):
            plt.subplot(5, 5, i + 1, title=f"predicted: {test_pred[i]}\ntrue: {test_true[i]}")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_valid[i], cmap='gray')
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        print(image.shape)
        image = tf.expand_dims(image, 0)
        with file_writer.as_default():
            tf.summary.image("Training data", image, step=0)



        
        # file_writer = tf.summary.create_file_writer(logdir+'/imgs')
        # with file_writer.as_default():
        #     # Don't forget to reshape.
        #     images = x_train[0:25]
        #     tf.summary.image("25 training data examples", images, max_outputs=25, step=0)
        
        
    images_predictions_cb = tf.keras.callbacks.LambdaCallback(on_train_end=log_images_predictions)


    
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
    initial_epochs = 5
    
    # %tensorboard --logdir ../logs/tensorbord/
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=initial_epochs, callbacks=[tensorboard_cb, cm_cb, images_predictions_cb])

    print("run tensorboard")
    os.system(f'tensorboard --logdir={logdir}')
    
    print(f"Model trained for {initial_epochs} epochs")
    print(f"Model summary: {model.summary()}")