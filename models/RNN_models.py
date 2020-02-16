from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D, MaxPooling2D
from keras.layers import AveragePooling2D,Permute,Reshape,LSTM,Lambda,GRU,Bidirectional,BatchNormalization,Concatenate
from keras import regularizers
from keras.optimizers import Adam
from utils import sharpe_ratio_loss,sharpe_ratio
import keras.backend as K
from keras.models import Model

def build_lstm_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = BatchNormalization()(reshape_inp)
    recurrent_layer = LSTM(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1)) (batch_norm)

    batch_norm_2 = BatchNormalization()(recurrent_layer)


    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_2)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model

def build_gru_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']

    input_shape = params['input_shape']
    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = BatchNormalization()(reshape_inp)
    recurrent_layer = GRU(units = units,
                        activation = activation,
                      kernel_regularizer=regularizers.l2(reg1)) (batch_norm)

    batch_norm_2 = BatchNormalization()(recurrent_layer)
    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_2)
    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])

    return model
