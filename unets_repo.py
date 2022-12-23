from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, Concatenate

### bloques

def conv_block(inputs, activation="relu", initializer="he_normal", num_filters=64):
    
    conv = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = initializer)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    conv = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = initializer)(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    
    return conv

def deconv_block(inputs, activation="relu", initializer="he_normal", num_filters=64,concat=None):
    
    up = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    merge = concatenate([up,concat], axis = 3)
    conv = conv_block(merge, activation=activation, initializer=initializer, num_filters=num_filters)
    
    return conv

def UNet(input_size = (256,256,1), activation = "relu", initializer = "he_normal"):
    
    # Input
    inputs = Input(input_size)
    
    # Encoder
    conv1 = conv_block(inputs,num_filters=64,activation=activation,initializer=initializer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1,num_filters=128,activation=activation,initializer=initializer)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2,num_filters=256,activation=activation,initializer=initializer)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3,num_filters=512,activation=activation,initializer=initializer)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, num_filters=1024, activation=activation, initializer=initializer)
 
    # Decoder
    deconv1 = deconv_block(conv5,num_filters=512,activation=activation,initializer=initializer,concat=conv4)
    deconv2 = deconv_block(deconv1,num_filters=256,activation=activation,initializer=initializer,concat=conv3)
    deconv3 = deconv_block(deconv2,num_filters=128,activation=activation,initializer=initializer,concat=conv2)
    deconv4 = deconv_block(deconv3,num_filters=64,activation=activation,initializer=initializer,concat=conv1)

    # Output
    output = Conv2D(1, 1, activation = 'sigmoid')(deconv4)

    model = Model(inputs, output, name = "U-Net")

    return model