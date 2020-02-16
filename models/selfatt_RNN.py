from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D
from keras.layers import MaxPooling2D,AveragePooling2D,Permute,Reshape,LSTM,Lambda,GRU,Bidirectional,BatchNormalization,Concatenate
from keras import regularizers
from keras.optimizers import Adam
from models.attention_layer import *
from utils import sharpe_ratio_loss,sharpe_ratio
import keras.backend as K
from keras.models import Model

def build_selfatt_gru_model(params):
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

    prob = SelfAttentionLayer(latent_dim=32,name='Att',kernel_regularizer=regularizers.l2(1e-4))(batch_norm)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,batch_norm])

    recurrent_layer = GRU(units = units,
                        activation = activation,
                      kernel_regularizer=regularizers.l2(1e-4)) (att)

    batch_norm_2 = BatchNormalization()(recurrent_layer)

    out = Dense(tickers, kernel_regularizer =regularizers.l2(1e-4)) (batch_norm_2)
    out = Activation('sigmoid',name='main_out')(out)

    ### output2 for MSE loss (return)
    # out2 = Dense(tickers,name='aux_out') (batch_norm_2)
    model = Model([input], [out])
    optimizer = Adam(lr = 1e-3)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])

    return model


def build_selfatt_lstm_model(params):
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

    # batch_norm = BatchNormalization()(reshape_inp)
    prob = SelfAttentionLayer(latent_dim=32,name='Att',kernel_regularizer=regularizers.l2(reg1))(reshape_inp)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,reshape_inp])

    recurrent_layer = LSTM(units = units,
    					activation = activation,
    				  kernel_regularizer=regularizers.l2(reg2)) (att)

    batch_norm_2 = BatchNormalization()(recurrent_layer)

    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_2)
    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])


    return model
