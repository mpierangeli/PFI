from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, multiply, add
from keras import backend as K


### bloques

def conv_block(inputs, activation="relu", initializer="he_normal", num_filters=64, dropout=False):
    
    conv = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = initializer)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    conv = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = initializer)(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    if dropout:
        conv = Dropout(dropout)(conv)
    
    return conv

def simple_conv_block(inputs, activation="relu", initializer="he_normal", num_filters=64, dropout=False):
    
    conv = Conv2D(num_filters, 3, padding = 'same', kernel_initializer = initializer)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    if dropout:
        conv = Dropout(dropout)(conv)
    
    return conv

def deconv_block(inputs, activation="relu", initializer="he_normal", num_filters=64,concat=None):
    
    up = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    merge = concatenate([up,concat], axis = 3)
    conv = conv_block(merge, activation=activation, initializer=initializer, num_filters=num_filters)
    
    return conv

def classic_deconv_block(inputs, activation="relu", initializer="he_normal", num_filters=64,concat=None):
    
    up = UpSampling2D((2, 2), data_format="channels_last")(inputs)
    up = Conv2D(num_filters, (2, 2), padding='same', kernel_initializer = initializer)(up)
    merge = concatenate([up,concat], axis = 3)
    conv = conv_block(merge, activation=activation, initializer=initializer, num_filters=num_filters)
    
    return conv

def res_deconv_block(inputs, activation="relu", initializer="he_normal", num_filters=64,concat=None):
    
    up = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    merge = concatenate([up,concat], axis = 3)
    conv = res_conv_block(merge, activation=activation, initializer=initializer, num_filters=num_filters)
    
    return conv

def attention_block(x, gating, num_filters):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = Conv2D(num_filters, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(num_filters, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(num_filters, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    
    return result_bn

def gating_signal(input, num_filters):

    x = Conv2D(num_filters, (1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def repeat_elem(tensor, rep):

    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)

def res_conv_block(x, num_filters=64,initializer="he_normal", activation="relu"):

    conv = Conv2D(num_filters, (3,3), padding='same', kernel_initializer = initializer)(x)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    conv = Conv2D(num_filters, (3,3), padding='same',kernel_initializer = initializer)(conv)
    conv = BatchNormalization()(conv)

    shortcut = Conv2D(num_filters, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, conv])
    res_path = Activation(activation)(res_path)    
    
    return res_path

### modelos

def UNet(input_size = (256,256,1), activation = "relu", initializer = "he_normal",num_filters=64, dropout=False):
    
    # Input
    inputs = Input(input_size)
    
    # Encoder
    conv1 = conv_block(inputs,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1,num_filters=num_filters*2,activation=activation,initializer=initializer,dropout=dropout)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2,num_filters=num_filters*4,activation=activation,initializer=initializer,dropout=dropout)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3,num_filters=num_filters*8,activation=activation,initializer=initializer,dropout=dropout)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, num_filters=num_filters*16, activation=activation, initializer=initializer)
 
    # Decoder
    deconv1 = deconv_block(conv5,num_filters=num_filters*8,activation=activation,initializer=initializer,concat=conv4)
    deconv2 = deconv_block(deconv1,num_filters=num_filters*4,activation=activation,initializer=initializer,concat=conv3)
    deconv3 = deconv_block(deconv2,num_filters=num_filters*2,activation=activation,initializer=initializer,concat=conv2)
    deconv4 = deconv_block(deconv3,num_filters=num_filters,activation=activation,initializer=initializer,concat=conv1)

    # Output
    output = Conv2D(1, 1)(deconv4)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    model = Model(inputs, output, name = "U-Net")

    return model

def classicUNet(input_size = (256,256,1), activation = "relu", initializer = "he_normal",num_filters=64, dropout=False):
    
    # Input
    inputs = Input(input_size)
    
    # Encoder
    conv1 = conv_block(inputs,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1,num_filters=num_filters*2,activation=activation,initializer=initializer,dropout=dropout)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2,num_filters=num_filters*4,activation=activation,initializer=initializer,dropout=dropout)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3,num_filters=num_filters*8,activation=activation,initializer=initializer,dropout=dropout)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, num_filters=num_filters*16, activation=activation, initializer=initializer)
 
    # Decoder
    deconv1 = classic_deconv_block(conv5,num_filters=num_filters*8,activation=activation,initializer=initializer,concat=conv4)
    deconv2 = classic_deconv_block(deconv1,num_filters=num_filters*4,activation=activation,initializer=initializer,concat=conv3)
    deconv3 = classic_deconv_block(deconv2,num_filters=num_filters*2,activation=activation,initializer=initializer,concat=conv2)
    deconv4 = classic_deconv_block(deconv3,num_filters=num_filters,activation=activation,initializer=initializer,concat=conv1)

    # Output
    output = Conv2D(1, 1)(deconv4)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    model = Model(inputs, output, name = "classic_U-Net")

    return model

def AttUnet(input_size = (256,256,1), activation = "relu", initializer = "he_normal",num_filters=64, dropout=False):
    
    inputs = Input(input_size)

    # Encoder
    conv1 = conv_block(inputs,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1,num_filters=num_filters*2,activation=activation,initializer=initializer,dropout=dropout)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2,num_filters=num_filters*4,activation=activation,initializer=initializer,dropout=dropout)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3,num_filters=num_filters*8,activation=activation,initializer=initializer,dropout=dropout)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, num_filters=num_filters*16, activation=activation, initializer=initializer)
 
    # Decoder
    gating1 = gating_signal(conv5, num_filters=num_filters*8)
    att1 = attention_block(conv4, gating1, num_filters=num_filters*8)
    deconv1 = deconv_block(conv5,num_filters=num_filters*8,activation=activation,initializer=initializer,concat=att1)
    gating2 = gating_signal(deconv1, num_filters=num_filters*4)
    att2 = attention_block(conv3, gating2, num_filters=num_filters*4)
    deconv2 = deconv_block(deconv1,num_filters=num_filters*4,activation=activation,initializer=initializer,concat=att2)
    gating3 = gating_signal(deconv2,num_filters=num_filters*2)
    att3 = attention_block(conv2, gating3,num_filters=num_filters*2)
    deconv3 = deconv_block(deconv2,num_filters=num_filters*2,activation=activation,initializer=initializer,concat=att3)
    gating4 = gating_signal(deconv3, num_filters=num_filters)
    att4 = attention_block(conv1, gating4, num_filters=num_filters)
    deconv4 = deconv_block(deconv3,num_filters=num_filters,activation=activation,initializer=initializer,concat=att4)
    
    output = Conv2D(1,1)(deconv4)
    output = BatchNormalization()(output)
    output = Activation('sigmoid')(output)

    model  = Model(inputs, output, name = "AttU-Net")
    
    return model

def ResAttUnet(input_size = (256,256,1), activation = "relu", initializer = "he_normal",num_filters=64,bn=False):
    
    inputs = Input(input_size)

    # Encoder
    conv1 = res_conv_block(inputs,num_filters=num_filters,activation=activation,initializer=initializer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = res_conv_block(pool1,num_filters=num_filters*2,activation=activation,initializer=initializer)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = res_conv_block(pool2,num_filters=num_filters*4,activation=activation,initializer=initializer)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = res_conv_block(pool3,num_filters=num_filters*8,activation=activation,initializer=initializer)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = res_conv_block(pool4, num_filters=num_filters*16, activation=activation, initializer=initializer)
 
    # Decoder
    gating1 = gating_signal(conv5, num_filters=num_filters*8)
    att1 = attention_block(conv4, gating1, num_filters=num_filters*8)
    deconv1 = res_deconv_block(conv5,num_filters=num_filters*8,activation=activation,initializer=initializer,concat=att1)
    gating2 = gating_signal(deconv1, num_filters=num_filters*4)
    att2 = attention_block(conv3, gating2, num_filters=num_filters*4)
    deconv2 = res_deconv_block(deconv1,num_filters=num_filters*4,activation=activation,initializer=initializer,concat=att2)
    gating3 = gating_signal(deconv2,num_filters=num_filters*2)
    att3 = attention_block(conv2, gating3,num_filters=num_filters*2)
    deconv3 = res_deconv_block(deconv2,num_filters=num_filters*2,activation=activation,initializer=initializer,concat=att3)
    gating4 = gating_signal(deconv3, num_filters=num_filters)
    att4 = attention_block(conv1, gating4, num_filters=num_filters)
    deconv4 = res_deconv_block(deconv3,num_filters=num_filters,activation=activation,initializer=initializer,concat=att4)
    
    output = Conv2D(1,1)(deconv4)
    if bn:
        output = BatchNormalization()(output) #mepa q la entrene sin esto
    output = Activation('sigmoid')(output)

    model  = Model(inputs, output, name = "AttU-Net")
    
    return model

def unet3plus(input_size = (256,256,1), activation = "relu", initializer = "he_normal",num_filters=64,dropout=False):

    inputs = Input(input_size)

    # Encoder
    conv1 = conv_block(inputs,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1,num_filters=num_filters*2,activation=activation,initializer=initializer,dropout=dropout)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2,num_filters=num_filters*4,activation=activation,initializer=initializer,dropout=dropout)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3,num_filters=num_filters*8,activation=activation,initializer=initializer,dropout=dropout)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = conv_block(pool4, num_filters=num_filters*16, activation=activation, initializer=initializer)

    # Decoder
    cat_channels = num_filters
    cat_blocks = 5 #capas de la red (1,2,4,8,16)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = MaxPooling2D(pool_size=(8, 8))(conv1)
    e1_d4 = simple_conv_block(e1_d4,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e2_d4 = MaxPooling2D(pool_size=(4, 4))(conv2)
    e2_d4 = simple_conv_block(e2_d4,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e3_d4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    e3_d4 = simple_conv_block(e3_d4,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e4_d4 = simple_conv_block(conv4,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e5_d4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)
    e5_d4 = simple_conv_block(e5_d4,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d4 = concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = simple_conv_block(d4,num_filters=upsample_channels,activation=activation,initializer=initializer,dropout=dropout)

    """ d3 """
    e1_d3 = MaxPooling2D(pool_size=(4, 4))(conv1)
    e1_d3 = simple_conv_block(e1_d3,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e2_d3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    e2_d3 = simple_conv_block(e2_d3,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e3_d3 = simple_conv_block(conv3,num_filters=upsample_channels,activation=activation,initializer=initializer,dropout=dropout)

    e4_d3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)
    e4_d3 = simple_conv_block(e4_d3,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e5_d3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(conv5)
    e5_d3 = simple_conv_block(e5_d3,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d3 = concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = simple_conv_block(d3,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    """ d2 """
    e1_d2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    e1_d2 = simple_conv_block(e1_d2,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e2_d2 = simple_conv_block(conv2,num_filters=upsample_channels,activation=activation,initializer=initializer,dropout=dropout)

    d3_d2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)
    d3_d2 = simple_conv_block(d3_d2,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d4_d2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)
    d4_d2 = simple_conv_block(d4_d2,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e5_d2 = UpSampling2D(size=(8, 8), interpolation='bilinear')(conv5)
    e5_d2 = simple_conv_block(e5_d2,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d2 = concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = simple_conv_block(d2,num_filters=upsample_channels,activation=activation,initializer=initializer,dropout=dropout)

    """ d1 """
    e1_d1 = simple_conv_block(conv1,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d2_d1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)
    d2_d1 = simple_conv_block(d2_d1,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d3_d1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)
    d3_d1 = simple_conv_block(d3_d1,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d4_d1 = UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)
    d4_d1 = simple_conv_block(d4_d1,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    e5_d1 = UpSampling2D(size=(16, 16), interpolation='bilinear')(conv5)
    e5_d1 = simple_conv_block(e5_d1,num_filters=num_filters,activation=activation,initializer=initializer,dropout=dropout)

    d1 = concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = simple_conv_block(d1,num_filters=upsample_channels,activation=activation,initializer=initializer,dropout=dropout)

    output = Conv2D(1,1)(d1)
    output = BatchNormalization()(output)
    output = Activation('sigmoid')(output)

    model  = Model(inputs, output, name = "UNet3plus")
    
    return model