
import keras.backend as K
from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D, MaxPooling2D,AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras import regularizers
from keras.optimizers import Adam,SGD
from keras.models import Model
from utils import sharpe_ratio_loss,sharpe_ratio

def bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)


def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return bn_relu(conv)

    return f


def bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    """

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

def short_cut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """

    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=regularizers.l2(0.001))(input)

    return Add()([shortcut, residual])

def residual_block(filters, repetitions,kernel_size=(3,3),strides=(2,2), is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = strides
            input = basic_block(filters=filters,kernel_size=kernel_size, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters,kernel_size=(3,3), init_strides=(1, 1), is_first_block_of_first_layer=False):

    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="glorot_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=kernel_size,
                                  strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=kernel_size)(conv1)
        return short_cut(input, residual)

    return f

def build_resnet_model(params):


    conv1_ksize = params['filters_1']
    conv1_nfilter = params['filters']

    kernel_size_1 = params['repetitions_1']
    kernel_size_2 = params['repetitions_3']
    kernel_size_3 = params['repetitions_5']
    kernel_size_4 = params['repetitions_7']


    num_filter_1 = params['filters_2']
    num_filter_2 = params['filters_3']
    num_filter_3 = params['filters_4']
    num_filter_4 = params['filters_5']


    reps_1 = params['repetitions']
    reps_2 = params['repetitions_2']
    reps_3 = params['repetitions_4']
    reps_4 = params['repetitions_6']

    conv2_nfilter = params['filters_6']


    regularized_coff_1 = params['l2']
    regularized_coff_2 = params['l2_1']
    regularized_coff_3 = params['l2_2']
    learning_rate = params['l2_3']
    input_shape = params['input_shape']
    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    conv1 = conv_bn_relu(filters=conv1_nfilter,kernel_size=(1,conv1_ksize),strides=(1,1),\
                         kernel_regularizer=regularizers.l2(regularized_coff_1)) (input)

    pool1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding="same")(conv1)

    out = residual_block(filters=num_filter_1, repetitions=reps_1 ,kernel_size=(1,kernel_size_1),\
                         strides=(1,2),is_first_layer=True) (pool1)

    out = residual_block(filters=num_filter_2, repetitions=reps_2,\
                         kernel_size=(1,kernel_size_2), strides=(1,2)) (out)

    out = residual_block(filters=num_filter_3, repetitions=reps_3,\
                         kernel_size=(1,kernel_size_3),strides=(1,2)) (out)

    out = residual_block(filters=num_filter_4, repetitions=reps_4,\
                         kernel_size=(1,kernel_size_4),strides=(1,2)) (out)

    out = bn_relu(out)

    conv2 = conv_bn_relu(filters=conv2_nfilter,kernel_size=(381,1),strides=(1,1),\
                     kernel_regularizer=regularizers.l2(regularized_coff_2),padding='valid') (out)

    out_shape = K.int_shape(conv2)
    out = AveragePooling2D(pool_size=(out_shape[1], out_shape[2]),
                                 strides=(1, 1))(conv2)

    out = Flatten()(out)

    out = Dense(tickers, kernel_regularizer =regularizers.l2(regularized_coff_3))(out)
    out = Activation('sigmoid')(out)


    model = Model([input], [out])
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])

    return model
