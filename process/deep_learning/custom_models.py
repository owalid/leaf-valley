import tensorflow as tf
import tensorflow.keras.layers as tfl

class CopyChannels(tf.keras.layers.Layer):
    """
    This layer copies channels from channel_start the number of channels given in channel_count.
    """
    def __init__(self,
                 channel_start=0,
                 channel_count=1,
                 **kwargs):
        self.channel_start=channel_start
        self.channel_count=channel_count
        super(CopyChannels, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.channel_count)
    
    def call(self, x):
        return x[:, :, :, self.channel_start:(self.channel_start+self.channel_count)]
        
    def get_config(self):
        config = {
            'channel_start': self.channel_start,
            'channel_count': self.channel_count
        }
        base_config = super(CopyChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def classic_cnn(input_shape, num_classes):
    return tf.keras.models.Sequential([
        tfl.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=input_shape),
        tfl.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        tfl.flatten(),
        tfl.GlobalAveragePooling2D(),
        tfl.Dense(512, activation='relu'),
        tfl.Dropout(0.5),
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
        tfl.Dropout(0.5),
        tfl.Dense(num_classes, activation='softmax')
    ])

def lab_process(input_shape, num_classes):
    '''
        Inspiration: https://github.com/joaopauloschuler/two-path-noise-lab-plant-disease
    '''
    inputs = tf.keras.Input(shape=input_shape)
    
    # layer copies channels from channel_start the number of channels given in channel_count.
    l_input = CopyChannels(0,1)(inputs)
    ab_input = CopyChannels(1,2)(inputs)
    
    # L processing
    l_proc = tfl.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="same")(l_input)
    l_proc = tfl.Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="same")(l_input)
    l_proc = tfl.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same")(l_input)
    l_proc = tfl.MaxPooling2D()(l_proc)

    # AB processing
    ab_proc = tfl.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="same")(ab_input)
    ab_proc = tfl.Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="same")(ab_input)
    ab_proc = tfl.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same")(ab_input)
    ab_proc = tfl.MaxPooling2D()(ab_proc)

    # LAB concatenation
    lab_model = tfl.concatenate([l_proc, ab_proc])
    lab_model = tfl.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(lab_model)
    lab_model = tfl.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(lab_model)
    lab_model = tfl.MaxPooling2D()(lab_model)
    
    # Implement inception block for lab concatenation
    
    # Block a
    # 1x1 convolution
    conv1x1 = tfl.Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(lab_model)
    
    # 3x3 convolution
    conv3x3 = tfl.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(lab_model)
    conv3x3 = tfl.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(conv3x3)
    
    # 5x5 convolution
    conv5x5 = tfl.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(lab_model)
    conv5x5 = tfl.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(conv5x5)
    
    # Max pooling
    max_pool = tfl.MaxPool2D(pool_size=(3,3), strides=(1,1), padding="same")(lab_model)
    max_pool = tfl.Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(max_pool)
    
    # Concatenate all the layers
    layers = tfl.concatenate([conv1x1, conv3x3, conv5x5, max_pool], axis=-1)
    
    # Block b
    # 1x1 convolution
    conv1x1 = tfl.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(layers)
    
    # 3x3 convolution
    conv3x3 = tfl.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(layers)
    conv3x3 = tfl.Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(conv3x3)
    
    # 5x5 convolution
    conv5x5 = tfl.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(layers)
    conv5x5 = tfl.Conv2D(filters=96, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(conv5x5)
    
    # Max pooling
    max_pool = tfl.MaxPool2D(pool_size=(3,3), strides=(1,1), padding="same")(layers)
    max_pool = tfl.Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(max_pool)
    
    # Concatenate all the layers
    layers = tf.keras.layers.concatenate([conv1x1, conv3x3, conv5x5, max_pool], axis=-1)
    lab_model = tfl.concatenate([lab_model, layers], axis=-1)

	# Flatten the output of the inception block
    lab_model = tfl.GlobalAveragePooling2D()(lab_model)
    lab_model = tfl.Dense(512, activation='relu')(lab_model)
    lab_model = tfl.Dropout(0.5)(lab_model)
    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    outputs = prediction_layer(lab_model)
    model = tf.keras.Model(inputs, outputs)
    return model
    