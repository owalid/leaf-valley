import os
import argparse as ap
from argparse import RawTextHelpFormatter
import sys
sys.path.append('../../utilities')
from utils import get_dataset
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K

def mobile_net_v2(input_shape):
    '''
        Define a tf.keras model for binary classification out of the MobileNetV2 model
        Arguments:
            image_shape -- Image width and height
        Returns:
            tf.keras.model
    '''
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    
    # freeze the base model by making it non trainable
    base_model.trainable = False 
    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # apply data augmentation to the inputs
    x = inputs
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 
    
    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(rate=0.2)(x)
    prediction_layer = tfl.Softmax()
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
    print(f"img_size: {img_size}")
    print("hf['classes']", hf['classes'])
    # model = mobile_net_v2(img_size)
    # base_learning_rate = 0.001
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    #             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #             metrics=['accuracy'])
    # initial_epochs = 5
    # history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)