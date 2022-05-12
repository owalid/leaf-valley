import tensorflow as tf
import tensorflow.keras.layers as tfl

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

def vgg_19(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the VGG19 model
        Arguments:
            image_shape -- Image width and height
            num_classes -- Number of classes
        Returns:
            tf.keras.model
    '''

    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
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


def alexnet(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the AlexNet model
        Arguments:
            image_shape -- Image width and height
            num_classes -- Number of classes
        Returns:
            tf.keras.model
    '''
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.alexnet.AlexNet(input_shape=input_shape, include_top=False, weights='imagenet')
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
    
    
def resnet_50(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the ResNet50 model
        Arguments:
            image_shape -- Image width and height
            num_classes -- Number of classes
        Returns:
            tf.keras.model
    '''
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
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

def inception_v3(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the inception_v3 model
        Arguments:
            image_shape -- Image width and height
            num_classes -- Number of classes
        Returns:
            tf.keras.model
    '''
    
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
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

def efficient_net(input_shape, num_classes):
    '''
        Define a tf.keras model for multiclasses classification out of the inception_v3 model
        Arguments:
            image_shape -- Image width and height
            num_classes -- Number of classes
        Returns:
            tf.keras.model
    '''
    
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
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