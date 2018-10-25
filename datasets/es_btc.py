import pathlib
import os

import h5py
import pandas as pd
import numpy as np

from datasets.base import Dataset

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'es_btc'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'btc.h5'

class ESBTCDataset(Dataset):
    """5 Emini S&P 500 Buy the Close"""
    
    def __init__(self):
        self.window_size = 36
        self.emb_size = 9
        self.step = 1
        self.forecast = 1
        
    def load_or_generate_data(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            self._download_and_process_esbtc()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]
    
    def _download_and_process_esbtc(self):
        
        PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        
        # Load Pandas Dataframe and add columns
        print('Loading Trading Data...')
        fd = pd.read_hdf('../data/processed/store.h5', key='cnn_data')
        fd['change'] = fd['close'] - fd['close'].shift(1)
        fd['cdl_sign'] = np.sign(fd['close'] - fd['open'])
        fd['cdl_body'] = np.absolute(fd['close'] - fd['open'])
        fd['cdl_ut'] = np.where(fd['cdl_sign'] > 0, fd['high'] - fd['close'], fd['high'] - fd['open'])
        fd['cdl_lt'] = np.where(fd['cdl_sign'] > 0, fd['open'] - fd['low'], fd['close'] - fd['low'])
        fd['cdl_rng'] = fd['high'] - fd['low']
        fd['cdl_hl'] = np.where(fd['low'] >= fd['low'].shift(), 1, 0) #higher low
        fd['cdl_lh'] = np.where(fd['high'] <= fd['high'].shift(), 1, 0) #lower high
        
        #Turn df columns into variables
        print('Processing Trading Data...')
        data = fd[81:]
        openp = data['open'].tolist()
        highp = data['high'].tolist()
        lowp = data['low'].tolist()
        closep = data['close'].tolist()
        emap = data['ema'].tolist()
        sin_time = data['sin_time'].tolist()
        cos_time = data['cos_time'].tolist()
        btc = data['btc'].tolist()
        stc = data['stc'].tolist()
        change = data['change'].tolist()
        cdl_sign = data['cdl_sign'].tolist()
        cdl_body = data['cdl_body'].tolist()
        cdl_ut = data['cdl_ut'].tolist()
        cdl_lt = data['cdl_lt'].tolist()
        cdl_rng = data['cdl_rng'].tolist()
        cdl_hl = data['cdl_hl'].tolist()
        cdl_lh = data['cdl_lh'].tolist()
        
        #Create stack of observations
        WINDOW = self.window_size #Number of bars in a trading day
        EMB_SIZE = self.emb_size
        STEP = self.step
        FORECAST = self.forecast

        X, Y = [], []
        for i in range(0, len(data)-WINDOW+1, STEP):
            try:
                o = openp[i:i+WINDOW]
                h = highp[i:i+WINDOW]
                l = lowp[i:i+WINDOW]
                c = closep[i:i+WINDOW]
                e = emap[i:i+WINDOW]
                ct = cos_time[i:i+WINDOW]
                st = sin_time[i:i+WINDOW]
        
                cng = change[i:i+WINDOW]
        
                _cdl_sign = cdl_sign[i:i+WINDOW]
                _cdl_body = cdl_body[i:i+WINDOW]
                _cdl_ut = cdl_ut[i:i+WINDOW]
                _cdl_lt = cdl_lt[i:i+WINDOW]
                _cdl_rng = cdl_rng[i:i+WINDOW]
                _cdl_hl = cdl_hl[i:i+WINDOW]
                _cdl_lh = cdl_lh[i:i+WINDOW]
        
        
                o = (np.array(o) - np.mean(o)) / np.std(o)
                h = (np.array(h) - np.mean(h)) / np.std(h)
                l = (np.array(l) - np.mean(l)) / np.std(l)
                c = (np.array(c) - np.mean(c)) / np.std(c)
                e = (np.array(e) - np.mean(e)) / np.std(e)
        
                _cng = (np.array(cng) - np.mean(cng)) / np.std(cng)

                x_i = closep[i:i+WINDOW]
                y_i = closep[(i+WINDOW-1)+FORECAST]
        

                if btc[i+WINDOW-1] > 0:
                    y_i = [1, 0]
                else:
                    y_i = [0, 1]
        
                x_i = np.column_stack((o, h, l, c, e, ct, st, cng, _cdl_hl))
        
            except Exception as e:
                break

            #only add if 1pt body and close on high
            if (closep[i+WINDOW-1] == highp[i+WINDOW-1]) and (closep[i+WINDOW-1]-openp[i+WINDOW-1]>=1):
                X.append(x_i)
                Y.append(y_i)
                
        p = int(len(X) * 0.9)
        X, Y = np.array(X), np.array(Y)
        x_train = X[0:p]
        y_train = Y[0:p]
        x_test = X[p:]
        y_test = Y[p:]
        
        print('Saving to HDF5...')
        
        with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
            f.create_dataset('x_train', data=x_train, dtype='u1', compression='lzf')
            f.create_dataset('y_train', data=y_train, dtype='u1', compression='lzf')
            f.create_dataset('x_test', data=x_test, dtype='u1', compression='lzf')
            f.create_dataset('y_test', data=y_test, dtype='u1', compression='lzf')
            
        print('ES BTC data downloaded and processed')
        

if __name__ == '__main__':
    data = ESBTCDataset()
    data.load_or_generate_data()
    #print(data)
    print(data.x_train.shape, data.y_train.shape)
    print(data.x_test.shape, data.y_test.shape)