import argparse
from preprocess_data import *
from utils import load_config_file,MyEncoder
from models.addatt_RNN import *
from models.RNN_models import *
from models.resnet import *
from sklearn.model_selection import TimeSeriesSplit

class DELAFO:
    def __init__(self,model,path_data):
        self.model = model
        self.X,self.y,self.tickers = prepair_data(path_data,window_x=64,window_y=19)


    def write_log(self,history,path_dir,name_file):
        his = history.history
        if os.path.exists(path_dir)==False:
            os.mkdirs(path_dir)
        with open(os.path.join(path_dir,name_file), 'w') as outfile:
            json.dump(his, outfile,cls=MyEncoder, indent=2)
        print("write file log at %s"%(path))

    def train_model(self,n_fold,batch_size,epochs):
        tscv = TimeSeriesSplit(n_splits=n_fold)
        for train_index, test_index in tscv.split(self.X):
            X_tr, X_val = self.X[train_index], self.X[test_index[range(18,len(test_index),19)]]
            y_tr, y_val = self.y[train_index], self.y[test_index[range(18,len(test_index),19)]]

            his = self.model.fit(X_tr, y_tr, batch_size=batch_size, epochs= epochs,
                          validation_data=(X_val,y_val))
            self.write_log(his,'./logs',"log_%d.txt"%(test_index[-1]))



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    list_model = {'ResNet':"./config/resnet_hyper_params.json",
                    'GRU': "./config/gru_hyper_params.json",
                    'LSTM':"./config/lstm_hyper_params.json",
                    'AA_GRU':"./config/gru_hyper_params.json",
                    'AA_LSTM':"./config/lstm_hyper_params.json",
                    'SA_GRU':"./config/gru_hyper_params.json",
                    'SA_LSTM':"./config/lstm_hyper_params.json"}

    parser.add_argument('--data_path', type=str, help='Input dir for data')
    parser.add_argument('--model', choices=[m for m in list_model], default='AA_GRU')
    args = parser.parse_args()

    if args.model == "ResNet":
        hyper_params = load_config_file(list_model[args.model])
        model = build_resnet_model(hyper_params)
    elif args.model == "GRU":
        hyper_params = load_config_file(list_model[args.model])
        model = build_gru_model(hyper_params)
    elif args.model == "LSTM":
        hyper_params = load_config_file(list_model[args.model])
        model = build_lstm_model(hyper_params)
    elif args.model == "AA_GRU":
        hyper_params = load_config_file(list_model[args.model])
        model = build_add_att_gru_model(hyper_params)
    elif args.model == "AA_LSTM":
        hyper_params = load_config_file(list_model[args.model])
        model = build_add_att_lstm_model(hyper_params)
    elif args.model == "SA_GRU":
        hyper_params = load_config_file(list_model[args.model])
        model = build_selfatt_gru_model(hyper_params)
    else:
        hyper_params = load_config_file(list_model[args.model])
        model = build_selfatt_lstm_model(hyper_params)

    delafo = DELAFO(model,args.data_path)
    delafo.train_model(n_fold=10,batch_size=16,epochs=200)
