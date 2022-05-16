import tensorflow as tf
import tensorflow.keras.layers as tfl

def classic_cnn(input_shape, num_classes):
    return tf.keras.models.Sequential([
        tfl.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=input_shape),
        tfl.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        tfl.flatten(),
        tfl.GlobalAveragePooling2D(),
        tfl.Dense(512, activation='relu'),
        tfl.dropout(0.5),
        tf.Dense(num_classes, activation='softmax')
    ])

def alexnet(input_shape, num_classes):
    return tf.keras.models.Sequential([
        tfl.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        tfl.BatchNormalization(),
        tfl.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tfl.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        tfl.BatchNormalization(),
        tfl.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tfl.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tfl.BatchNormalization(),
        tfl.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tfl.BatchNormalization(),
        tfl.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tfl.BatchNormalization(),
        tfl.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tfl.Flatten(),
        tfl.GlobalAveragePooling2D(),
        tfl.Dense(512, activation='relu'),
        tfl.dropout(0.5),
        tfl.Dense(num_classes, activation='softmax')
    ])


def vgg_16(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the VGG16 model
        Arguments:
            image_shape -- Image width and height
            num_classes -- Number of classes
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