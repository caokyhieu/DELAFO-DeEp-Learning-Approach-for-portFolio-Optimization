import argparse
from preprocess_data import *
from utils import *
from models.addatt_RNN import *
from models.attention_layer import *
from models.RNN_models import *
from models.selfatt_RNN import *
from models.resnet import *
from sklearn.model_selection import TimeSeriesSplit
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
class DELAFO:
    def __init__(self,model_name,model,X,y,tickers,timesteps_input=64,timesteps_output=19):
        self.model_name = model_name
        self.model = model
        self.X,self.y,self.tickers = X,y,tickers
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output

    @classmethod
    def from_existing_config(cls,path_data,model_name,model_config_path,timesteps_input=64,timesteps_output=19):

        X,y,tickers = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)

        if model_name == "ResNet":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])

            model = build_resnet_model(hyper_params)
        elif model_name == "GRU":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])

            model = build_gru_model(hyper_params)
        elif model_name == "LSTM":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])

            model = build_lstm_model(hyper_params)
        elif model_name == "AA_GRU":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])
            print(hyper_params)
            model = build_add_att_gru_model(hyper_params)
        elif model_name == "AA_LSTM":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])
            model = build_add_att_lstm_model(hyper_params)
        elif model_name == "SA_GRU":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])
            model = build_selfatt_gru_model(hyper_params)
        elif model_name == "SA_LSTM":
            hyper_params = load_config_file(model_config_path[model_name])
            hyper_params['input_shape'] = (X.shape[1],X.shape[2],X.shape[3])
            model = build_selfatt_lstm_model(hyper_params)
        model.name = model_name

        return cls(model_name,model,X,y,tickers,timesteps_input,timesteps_output)

    @classmethod
    def from_saved_model(cls,path_data,model_path,timesteps_output):
        '''  If you load pretrain model with new custom layer, you should put it in custom_objects
            below.
        '''
        model = load_model(model_path,custom_objects={"AdditiveAttentionLayer":AdditiveAttentionLayer,
                                                        "SelfAttentionLayer":SelfAttentionLayer,
                                                        "sharpe_ratio_loss":sharpe_ratio_loss,
                                                        "sharpe_ratio":sharpe_ratio})
        model_name = model.name
        input_shape = K.int_shape(model.input)
        timesteps_input = input_shape[2]
        X,y,tickers = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
        return cls(model_name,model,X,y,tickers,timesteps_input,timesteps_output)

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
        self.visualize_log('./logs',self.model_name)

    def save_model(self,path_dir="pretrain_model"):
        if os.path.exists(os.path.join(path_dir,self.model_name))==False:
            os.makedirs(os.path.join(path_dir,self.model_name))
        ver = list(map(lambda x: int(x.split('.')[0]),[file for file in os.listdir(os.path.join(path_dir,self.model_name)) if file.endswith('.h5')]))
        if len(ver)>0:
            ver = np.max(ver) + 1
        else:
            ver = 0
        self.model.save(os.path.join(path_dir,self.model_name,str(ver) + '.h5'))
        print("Model saved at %s" % os.path.join(path_dir,self.model_name))

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

    def visualize_log(self,path_folder,model_name):
        n_cols = 6
        n_rows = 1
        fig,axes = plt.subplots(ncols = n_cols,figsize=(20,3))
        path_files = [os.path.join(path_folder,model_name,file) for file in os.listdir(os.path.join(path_folder,model_name)) if os.path.isfile(os.path.join(path_folder,model_name,file))]
        for i,path in enumerate(path_files[-6:]):

            with open(path) as f:
                history = json.loads(f.read())

            axes[i].plot(history['sharpe_ratio'][50:])
            axes[i].plot(history['val_sharpe_ratio'][50:])
            axes[i].set_ylabel('Sharpe_ratio')
            axes[i].set_xlabel('Epoch')
            axes[i].legend(['Train', 'Test'], loc='upper left')
        new_path = os.path.join('/'.join(path_folder.split('/')[:-1]),'plot',model_name)
        if os.path.exists(new_path)==False:
            os.makedirs(new_path)
        plt.savefig(os.path.join(new_path,'1.png'))


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
    parser.add_argument('--load_pretrained', type=bool, default=False,help='Load pretrain model')
    parser.add_argument('--model_path', type=str, default='',help='Path to pretrain model')
    parser.add_argument('--timesteps_input', type=int, default=64,help='timesteps (days) for input data')
    parser.add_argument('--timesteps_output', type=int, default=19,help='Timesteps (days) for output data ')
    args = parser.parse_args()

    if args.load_pretrained == False:
        delafo = DELAFO.from_existing_config(args.data_path,args.model,model_config_path,args.timesteps_input,args.timesteps_output)
        delafo.train_model(n_fold=10,batch_size=16,epochs=300)
        delafo.save_model()
    else:
        delafo = DELAFO.from_saved_model(args.data_path,args.model_path,args.timesteps_output)
        delafo.train_model(n_fold=10,batch_size=16,epochs=300)
        delafo.save_model()
