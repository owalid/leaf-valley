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
    inputs = tf.keras.Input(shape=input_shape)
    
    model = tfl.Conv2D(filters=256, kernel_size=(6,6), strides=(1,1), padding="same")(inputs)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool2D(pool_size=(16,16), strides=(16,16), padding='same')(model)
    
    model = tfl.Conv2D(filters=128, kernel_size=(4,4), strides=(1,1), padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool2D(pool_size=(8,8), strides=(8,8), padding='same')(model)
    
    model = tfl.Conv2D(filters=96, kernel_size=(2,2), strides=(1,1), padding="same")(model)
    model = tfl.ReLU()(model)
    model = tfl.MaxPool2D(pool_size=(4,4), strides=(4,4), padding='same')(model)
    
    model = tfl.Flatten()(model)
    model = tfl.Dense(512, activation='relu')(model)
    model = tfl.Dropout(0.5)(model)
    
    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    outputs = prediction_layer(model)
    model = tf.keras.Model(inputs, outputs)
    
    return model

def alexnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    model = tfl.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape)(inputs)
    model = tfl.BatchNormalization()(model)
    
    model = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2))(model)
    model = tfl.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(model)
    model = tfl.BatchNormalization()(model)
    
    model = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2))(model)
    model = tfl.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(model)
    model = tfl.BatchNormalization()(model)
    
    model = tfl.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(model)
    model = tfl.BatchNormalization()(model)
    
    model = tfl.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(model)
    model = tfl.BatchNormalization()(model)
    
    model = tfl.MaxPool2D(pool_size=(3,3), strides=(2,2))(model)
    model = tfl.Flatten()(model)
    model = tfl.Dense(512, activation='relu')(model)
    model = tfl.Dropout(0.5)(model)

    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    outputs = prediction_layer(model)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# LAB HSV models
    
def common_model_hsv_lab(model, inputs, num_classes):
    # Implement inception block for lab concatenation
    
    # Block a
    # 1x1 convolution
    conv1x1 = tfl.Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(model)
    
    # 3x3 convolution
    conv3x3 = tfl.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(model)
    conv3x3 = tfl.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(conv3x3)
    
    # 5x5 convolution
    conv5x5 = tfl.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(model)
    conv5x5 = tfl.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(conv5x5)
    
    # Max pooling
    max_pool = tfl.MaxPool2D(pool_size=(3,3), strides=(1,1), padding="same")(model)
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
    model = tfl.concatenate([model, layers], axis=-1)

	# Flatten the output of the inception block
    model = tfl.GlobalAveragePooling2D()(model)
    model = tfl.Dense(512, activation='relu')(model)
    model = tfl.Dropout(0.5)(model)
    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    outputs = prediction_layer(model)
    model = tf.keras.Model(inputs, outputs)
    return model
    
    

def hsv_process(input_shape, num_classes):
    '''
        Inspiration: https://github.com/joaopauloschuler/two-path-noise-lab-plant-disease
        
        SCHEMA:
        H   SV
        |   |
    '''
    inputs = tf.keras.Input(shape=input_shape)
    
    # layer copies channels from channel_start the number of channels given in channel_count.
    h_input = CopyChannels(0,1)(inputs)
    sv_input = CopyChannels(1,2)(inputs)
    
    # H processing
    h_proc = tfl.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="same")(h_input)
    h_proc = tfl.Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="same")(h_input)
    h_proc = tfl.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same")(h_input)
    h_proc = tfl.MaxPooling2D()(h_proc)

    # SV processing
    sv_proc = tfl.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="same")(sv_input)
    sv_proc = tfl.Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="same")(sv_input)
    sv_proc = tfl.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same")(sv_input)
    sv_proc = tfl.MaxPooling2D()(sv_proc)

    # LAB concatenation
    hsv_model = tfl.concatenate([h_proc, sv_proc])
    hsv_model = tfl.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(hsv_model)
    hsv_model = tfl.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(hsv_model)
    hsv_model = tfl.MaxPooling2D()(hsv_model)
    
    model = common_model_hsv_lab(hsv_model, inputs, num_classes)
    return model 

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
    
    model = common_model_hsv_lab(lab_model, inputs, num_classes)
    return model


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    """
        Utility function to apply conv + BN.
    """
    x = tfl.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, name=name)(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.Activation('relu')(x)
    return x

def lab_inceptionv3_process(input_shape, num_classes):
    '''
        Inspiration: https://github.com/joaopauloschuler/two-path-noise-lab-plant-disease
    '''
    inputs = tf.keras.Input(shape=input_shape)
    
    l_ratio = 0.2
    ab_ratio = 0.8
    
    # layer copies channels from channel_start the number of channels given in channel_count.
    l_input = CopyChannels(0,1)(inputs)
    ab_input = CopyChannels(1,2)(inputs)
    
    # L processing
    l_proc = tfl.Conv2D(filters=int(l_ratio*512), kernel_size=(21,21), strides=(2,2), padding="same")(l_input)
    l_proc = tfl.BatchNormalization()(l_proc)
    l_proc = tfl.Activation('relu')(l_proc)
    
    l_proc = tfl.Conv2D(filters=int(l_ratio*256), kernel_size=(11,11), strides=(1,1), padding="same")(l_input)
    l_proc = tfl.BatchNormalization()(l_proc)
    l_proc = tfl.Activation('relu')(l_proc)
    
    l_proc = tfl.Conv2D(filters=int(l_ratio*128), kernel_size=(5,5), strides=(1,1), padding="same")(l_input)
    l_proc = tfl.BatchNormalization()(l_proc)
    l_proc = tfl.Activation('relu')(l_proc)
    
    l_proc = tfl.Conv2D(filters=int(l_ratio*96), kernel_size=(3,3), strides=(1,1), padding="same")(l_input)
    l_proc = tfl.BatchNormalization()(l_proc)
    l_proc = tfl.Activation('relu')(l_proc)
    
    l_proc = tfl.MaxPooling2D()(l_proc)

    # AB processing
    ab_proc = tfl.Conv2D(filters=int(512*ab_ratio), kernel_size=(21,21), strides=(2,2), padding="same")(ab_input)
    ab_proc = tfl.BatchNormalization()(ab_proc)
    ab_proc = tfl.Activation('relu')(ab_proc)
    
    ab_proc = tfl.Conv2D(filters=int(256*ab_ratio), kernel_size=(11,11), strides=(1,1), padding="same")(ab_input)
    ab_proc = tfl.BatchNormalization()(ab_proc)
    ab_proc = tfl.Activation('relu')(ab_proc)
    
    ab_proc = tfl.Conv2D(filters=int(128*ab_ratio), kernel_size=(5,5), strides=(1,1), padding="same")(ab_input)
    ab_proc = tfl.BatchNormalization()(ab_proc)
    ab_proc = tfl.Activation('relu')(ab_proc)
    
    ab_proc = tfl.Conv2D(filters=int(96*ab_ratio), kernel_size=(3,3), strides=(1,1), padding="same")(ab_input)
    ab_proc = tfl.BatchNormalization()(ab_proc)
    ab_proc = tfl.Activation('relu')(ab_proc)
    
    ab_proc = tfl.MaxPooling2D()(ab_proc)

    # LAB concatenation
    x = tfl.concatenate([l_proc, ab_proc])
    x = tfl.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(x)
    x = tfl.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
    
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = tfl.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = tfl.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = tfl.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = tfl.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = tfl.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = tfl.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = tfl.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tfl.concatenate([branch3x3, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed3')
    
        
    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = tfl.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = tfl.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                            axis=3,
                            name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = tfl.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(
                                                x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = tfl.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                            axis=3,
                            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = tfl.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = tfl.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                            axis=3,
                            name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = tfl.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tfl.concatenate([branch3x3, branch7x7x3, branch_pool],
                            axis=3,
                            name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = tfl.concatenate([branch3x3_1, branch3x3_2],
                                    axis=3,
                                    name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = tfl.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                        axis=3)

        branch_pool = tfl.AveragePooling2D((3, 3),
                                            strides=(1, 1),
                                            padding='same')(
                                                x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = tfl.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                            axis=3,
                            name='mixed' + str(9 + i))
        
    
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dense(512, activation='relu')(x)
    x = tfl.Dropout(0.5)(x)
    prediction_layer = tfl.Dense(num_classes, activation='softmax')
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())
    
    return model
    

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
    # 'CONVNEXT': {
    #     'base': tf.keras.applications.convnext,
    #     'preprocess_input': tf.keras.applications.convnext.preprocess_input
    # },
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
    'LAB_INCEPTIONV3_PROCESS': {
        'base': lab_inceptionv3_process,
        'preprocess_input': None
    },
    'HSV_PROCESS': {
        'base': hsv_process,
        'preprocess_input': None
    },
    'GOOGLE/VIT-BASE-PATCH16': {
        'model_id': 'google/vit-base-patch16-224-in21k',
        'is_hugging_face': True,
        'preprocess_input': None
    },
    'GOOGLE/VIT-BASE-PATCH32': {
        'model_id': 'google/vit-base-patch32-384',
        'is_hugging_face': True,
        'preprocess_input': None
    },
    'GOOGLE/VIT-LARGE-PATCH16': {
        'model_id': 'google/vit-large-patch16-224',
        'is_hugging_face': True,
        'preprocess_input': None
    },
    'GOOGLE/VIT-LARGE-PATCH32': {
        'model_id': 'google/vit-large-patch32-224-in21k',
        'is_hugging_face': True,
        'preprocess_input': None
    },
    'FACEBOOK/CONVNEXT-BASE': {
        'model_id': 'facebook/convnext-base-224-22k-1k',
        'is_hugging_face': True,
        'preprocess_input': None
    },
    'FACEBOOK/CONVNEXT-LARGE': {
        'model_id': 'facebook/convnext-large-224',
        'is_hugging_face': True,
        'preprocess_input': None
    },
    'FACEBOOK/CONVNEXT-XLARGE': {
        'model_id': 'facebook/convnext-xlarge-224-22k-1k',
        'is_hugging_face': True,
        'preprocess_input': None
    }
}