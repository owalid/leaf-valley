import tensorflow as tf
import tensorflow.keras.layers as tfl

# https://towardsdatascience.com/background-removal-with-u%C2%B2-net-2819b8e77078
# https://medium.com/analytics-vidhya/blur-or-change-background-of-images-using-machine-learning-with-tensorflow-f7dab3ddab6f

# def u_net2(input_shape):
#     '''
#         Create u2_net model with keras for segmentation of images of leaves.
#     '''
    
#     inputs = tf.keras.Input(shape=input_shape)
    
    
    
class automaticmaplabelling():
    def __init__(self, modelPath, width=512,height=512,channels=3):
        self.modelPath=modelPath
        self.IMG_WIDTH=width
        self.IMG_HEIGHT=height
        self.IMG_CHANNELS=channels
        self.model = self.U_net()
        
        make_dir(modelPath)
        
 

    def U_net(self):
        # Build U-Net model
        inputs = tfl.Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        
        c1 = tfl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tfl.Dropout(0.1)(c1)
        c1 = tfl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tfl.MaxPooling2D((2, 2))(c1)

        c2 = tfl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tfl.Dropout(0.1)(c2)
        c2 = tfl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tfl.MaxPooling2D((2, 2))(c2)

        c3 = tfl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tfl.Dropout(0.2)(c3)
        c3 = tfl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tfl.MaxPooling2D((2, 2))(c3)

        c4 = tfl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tfl.Dropout(0.2)(c4)
        c4 = tfl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tfl.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tfl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tfl.Dropout(0.3)(c5)
        c5 = tfl.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = tfl.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tfl.concatenate([u6, c4])
        c6 = tfl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tfl.Dropout(0.2)(c6)
        c6 = tfl.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tfl.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tfl.concatenate([u7, c3])
        c7 = tfl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tfl.Dropout(0.2)(c7)
        c7 = tfl.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tfl.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tfl.concatenate([u8, c2])
        c8 = tfl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tfl.Dropout(0.1)(c8)
        c8 = tfl.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tfl.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tfl.concatenate([u9, c1], axis=3)
        c9 = tfl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tfl.Dropout(0.1)(c9)
        c9 = tfl.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tfl.Conv2D(3, (1, 1), activation='sigmoid', name='seg')(c9)

        model = tfl.Model(inputs=[inputs], outputs=[outputs])

        losses = {'seg': 'binary_crossentropy'}

        metrics = {'seg': ['acc']}

        model.compile(optimizer='adam', loss=losses, metrics=metrics)
        model.load_weights(self.modelPath)

        return model