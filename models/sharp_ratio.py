
import keras.backend as K

def sharpe_ratio_loss(y_true, y_pred):
    epsilon = 1e-6
    max_bound = 1 - epsilon
    constraint_value = 3e-3
    y_pred_reshape = K.expand_dims(y_pred, axis=-1)
    z = y_true * y_pred_reshape
    z = K.sum(z, axis=1)

    sum_w = K.clip(K.sum(y_pred_reshape, axis=1), epsilon, 381* max_bound)
    ## constraint for number of tickers
#     num_constraint = C * (K.sum(y_pred) - 50)*(K.sum(y_pred)-10)
    rate = z/sum_w
    sharpeRatio = K.mean(rate, axis = 1)/K.maximum(K.std(rate, axis=1),epsilon)
    constraint =  K.sum((1.6 - y_pred) * y_pred,axis=1)
    return K.mean(constraint_value* constraint - sharpeRatio)

def sharpe_ratio(y_true, y_pred):
    epsilon = 1e-6
    y_pred_reshape = K.expand_dims(y_pred, axis=-1)
    y_pred_reshape= K.cast(K.greater(K.clip(y_pred_reshape, 0, 1), 0.5), K.floatx())

    z = y_true * y_pred_reshape
    z = K.sum(z, axis=1)
    sum_w = K.clip(K.sum(y_pred_reshape, axis=1), epsilon, 381)
    rate = z/sum_w
    sharpeRatio = K.mean(rate, axis = 1)/K.maximum(K.std(rate, axis=1),epsilon)

    return K.mean(sharpeRatio)
