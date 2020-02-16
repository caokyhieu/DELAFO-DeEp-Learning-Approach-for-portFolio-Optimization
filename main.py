import argparse
from preprocess_data import *
from utils import load_config_file,MyEncoder
from models.addatt_RNN import *
from models.RNN_models import *
from models.selfatt_RNN import *
from models.resnet import *
from sklearn.model_selection import TimeSeriesSplit
import os
import json
import numpy as np

class DELAFO:
    def __init__(self,path_data,model_name,model_config_path,timesteps_input=64,timesteps_output=19):

        self.model_name = model_name
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
        self.X,self.y,self.tickers = prepair_data(path_data,window_x=self.timesteps_input,window_y=self.timesteps_output)

        if self.model_name == "ResNet":
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])

            model = build_resnet_model(hyper_params)
        elif self.model_name == "GRU":
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])

            model = build_gru_model(hyper_params)
        elif self.model_name == "LSTM":
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])

            model = build_lstm_model(hyper_params)
        elif self.model_name == "AA_GRU":
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])
            print(hyper_params)
            model = build_add_att_gru_model(hyper_params)
        elif self.model_name == "AA_LSTM":
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])
            model = build_add_att_lstm_model(hyper_params)
        elif self.model_name == "SA_GRU":
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])
            model = build_selfatt_gru_model(hyper_params)
        else:
            hyper_params = load_config_file(model_config_path[self.model_name])
            hyper_params['input_shape'] = (self.X.shape[1],self.X.shape[2],self.X.shape[3])
            model = build_selfatt_lstm_model(hyper_params)

        self.model = model



    def write_log(self,history,path_dir,name_file):
        his = history.history
        if os.path.exists(path_dir)==False:
            os.makedirs(path_dir)
        with open(os.path.join(path_dir,name_file), 'w') as outfile:
            json.dump(his, outfile,cls=MyEncoder, indent=2)
        print("write file log at %s"%(os.path.join(path_dir,name_file)))

    def train_model(self,n_fold,batch_size,epochs):
        tscv = TimeSeriesSplit(n_splits=n_fold)
        for train_index, test_index in tscv.split(self.X):
            X_tr, X_val = self.X[train_index], self.X[test_index[range(self.timesteps_output-1,len(test_index),self.timesteps_output)]]
            y_tr, y_val = self.y[train_index], self.y[test_index[range(self.timesteps_output-1,len(test_index),self.timesteps_output)]]

            his = self.model.fit(X_tr, y_tr, batch_size=batch_size, epochs= epochs,validation_data=(X_val,y_val))
            mask_tickers = self.predict_portfolio(X_val)
            print('Sharpe ratio of this portfolio: %s' % str([self.calc_sharpe_ratio(mask_tickers[i],y_val[i]) for i in range(len(y_val))]))

            self.write_log(his,'./logs/%s' % self.model_name,"log_%d.txt"%(test_index[-1]))

    def predict_portfolio(self,X):
        results = self.model.predict(X)
        mask_tickers = results>0.5
        print("There are total %d samples to predict" % len(results))
        for i in range(len(mask_tickers)):
            print('Sample %d : [ %s ]' % (i, ' '.join([self.tickers[j] for j in range(len(self.tickers)) if mask_tickers[i][j]==1])))

        return mask_tickers

    def calc_sharpe_ratio(self,weight,y):
        """Here y is the daily return have the shape (tickers,days)
        weight have the shape (tickers,)"""
        epsilon = 1e-6
        weights = np.round(weight)
        sum_w = np.clip(weights.sum(),epsilon,y.shape[0])
        norm_weight = weights/sum_w
        port_return = norm_weight.dot(y).squeeze()
        mean = np.mean(port_return)
        std = np.maximum(np.std(port_return),epsilon)
        return np.sqrt(self.timesteps_output) * mean/std




if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    ## path for config_file of each model
    model_config_path = {'ResNet':"./config/resnet_hyper_params.json",
                        'GRU': "./config/gru_hyper_params.json",
                        'LSTM':"./config/lstm_hyper_params.json",
                        'AA_GRU':"./config/gru_hyper_params.json",
                        'AA_LSTM':"./config/lstm_hyper_params.json",
                        'SA_GRU':"./config/gru_hyper_params.json",
                        'SA_LSTM':"./config/lstm_hyper_params.json"}

    parser.add_argument('--data_path', type=str, help='Input dir for data')
    parser.add_argument('--model', choices=[m for m in model_config_path], default='AA_GRU')
    parser.add_argument('--timesteps_input', type=int, default=64,help='timesteps (days) for input data')
    parser.add_argument('--timesteps_output', type=int, default=19,help='Timesteps (days) for output data ')
    args = parser.parse_args()


    delafo = DELAFO(args.data_path,args.model,model_config_path,timesteps_input=64,timesteps_output=19)
    delafo.train_model(n_fold=10,batch_size=16,epochs=200)
