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
