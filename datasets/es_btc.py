import pathlib
import os

import h5py
import pandas as pd
import numpy as np

from datasets.base import Dataset

RAW_DATA_PATH = Dataset.data_dirname() / 'processed' / 'store.h5'
DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'es_btc'
#PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'btc.h5'

class ESBTCDataset(Dataset):
    """5 Emini S&P 500 Buy the Close"""
    
    def __init__(self, window_size: int=36, use_close: bool=True, use_cng: bool=False, use_time: bool=True, use_ema: bool=True,
                use_ohl: bool=False, use_cdl: bool=False, use_dir: bool=False, use_ema5: bool=False, use_ema3: bool=False):
        self.window_size = window_size
        
        self.use_close = use_close
        self.use_cng = use_cng
        self.use_time = use_time
        self.use_ema = use_ema
        self.use_ema5 = use_ema5
        self.use_ema3 = use_ema3
        self.use_ohl = use_ohl
        self.use_cdl = use_cdl
        self.use_dir = use_dir
        
        if self.use_close and self.use_cng:
            e_size = 2
        elif self.use_cng:
            e_size = 1
        else:
            e_size = 1
            self.use_close = True
        
        if self.use_time:
            e_size += 2
        if self.use_ema:
            e_size += 1
        if self.use_ohl:
            e_size += 3
        if self.use_cdl:
            e_size += 4
        if self.use_dir:
            e_size += 2
        if self.use_ema5:
            e_size += 1
        if self.use_ema3:
            e_size += 1
                 
        self.emb_size = e_size
        self.step = 1
        self.forecast = 1
        self.num_classes = 2
        self.input_shape = (self.window_size, self.emb_size)
        self.output_shape = (self.num_classes,)
        
    @property
    def data_filename(self):
        return DATA_DIRNAME / f'ws_{self.window_size}_close_{self.use_close}_cng_{self.use_cng}_t_{self.use_time}_ema_{self.use_ema}_ohl_{self.use_ohl}_cdl_{self.use_cdl}_dir_{self.use_dir}_ema5_{self.use_ema5}_ema3_{self.use_ema3}.h5'
        
    def load_or_generate_data(self):
        if not os.path.exists(self.data_filename):
            self._download_and_process_esbtc()
        with h5py.File(self.data_filename, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test = f['y_test'][:]
            
    def __repr__(self):
        return (
            'ESBTC Dataset\n'
            f'Window size: {self.window_size}\n'
            f'Num features: {self.emb_size}\n'
            )
    
    def _download_and_process_esbtc(self):
        
        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        
        # Load Pandas Dataframe and add columns
        print('Loading Trading Data...')
        fd = pd.read_hdf(RAW_DATA_PATH, key='cnn_data')
        fd['ema_5'] = fd['close'].ewm(span=5, min_periods=5).mean()
        fd['ema_3'] = fd['close'].ewm(span=3, min_periods=3).mean()
        fd['change'] = fd['close'] - fd['close'].shift(1)
        fd['cdl_hl'] = np.where(fd['low'] >= fd['low'].shift(), 1, 0) #higher low
        fd['cdl_lh'] = np.where(fd['high'] <= fd['high'].shift(), 1, 0) #lower high
        
        fd['cdl_body'] = (fd['close'] - fd['open']) / fd['close'] * 100
        fd['cdl_rng'] = fd['high'] - fd['low']
        fd['cdl_ut'] = np.where(fd['cdl_body'] > 0, fd['high'] - fd['close'], fd['high'] - fd['open'])
        fd['cdl_lt'] = np.where(fd['cdl_body'] > 0, fd['open'] - fd['low'], fd['close'] - fd['low'])
        fd['cdl_ut'] = np.where(fd['cdl_rng'] == 0, 0, fd['cdl_ut'] / fd['cdl_rng'])
        fd['cdl_lt'] = np.where(fd['cdl_rng'] == 0, 0, fd['cdl_lt'] / fd['cdl_rng'])
        fd['cdl_rng'] = (fd['cdl_rng'] / fd['low']) * 100

        
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
        #cdl_sign = data['cdl_sign'].tolist()
        cdl_body = data['cdl_body'].tolist()
        cdl_ut = data['cdl_ut'].tolist()
        cdl_lt = data['cdl_lt'].tolist()
        cdl_rng = data['cdl_rng'].tolist()
        cdl_hl = data['cdl_hl'].tolist()
        cdl_lh = data['cdl_lh'].tolist()
        ema5p = data['ema_5'].tolist()
        ema3p = data['ema_3'].tolist()
        
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
                e5 = ema5p[i:i+WINDOW]
                e3 = ema3p[i:i+WINDOW]
                ct = cos_time[i:i+WINDOW]
                st = sin_time[i:i+WINDOW]
        
                cng = change[i:i+WINDOW]
        
                #_cdl_sign = cdl_sign[i:i+WINDOW]
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
                e5 = (np.array(e5) - np.mean(e5)) / np.std(e5)
                e3 = (np.array(e3) - np.mean(e3)) / np.std(e3)
        
                _cng = (np.array(cng) - np.mean(cng)) / np.std(cng)

                x_i = closep[i:i+WINDOW]
                y_i = closep[(i+WINDOW-1)+FORECAST]
        

                if btc[i+WINDOW-1] > 0:
                    y_i = [1, 0]
                else:
                    y_i = [0, 1]
                    
                if self.use_close and self.use_cng:
                    args = (c, cng)
                elif self.use_cng:
                    args = (cng, )
                else:
                    args = (c, )
                    
                if self.use_time:
                    args += (ct, st)
                if self.use_ema:
                    args += (e, )
                if self.use_ema5:
                    args += (e5, )
                if self.use_ema3:
                    args += (e3, )
                if self.use_ohl:
                    args += (o, h, l)
                if self.use_cdl:
                    args += (_cdl_body, _cdl_ut, _cdl_lt, _cdl_rng)
                if self.use_dir:
                    args += (_cdl_hl, _cdl_lh)
        
                x_i = np.column_stack(args)
        
            except Exception as e:
                break

            #only add if 1pt body and close on high
            #if (closep[i+WINDOW-1] == highp[i+WINDOW-1]) and (closep[i+WINDOW-1]-openp[i+WINDOW-1]>=1):
            X.append(x_i)
            Y.append(y_i)
                
        p = int(len(X) * 0.9)
        X, Y = np.array(X), np.array(Y)
        x_train = X[0:p]
        y_train = Y[0:p]
        x_test = X[p:]
        y_test = Y[p:]
        
        print('Saving to HDF5...')
        
        with h5py.File(self.data_filename, 'w') as f:
            f.create_dataset('x_train', data=x_train, dtype='f4', compression='lzf')
            f.create_dataset('y_train', data=y_train, dtype='f4', compression='lzf')
            f.create_dataset('x_test', data=x_test, dtype='f4', compression='lzf')
            f.create_dataset('y_test', data=y_test, dtype='f4', compression='lzf')
            
        print('ES BTC data downloaded and processed')
        

if __name__ == '__main__':
    data = ESBTCDataset()
    data.load_or_generate_data()
    #print(data)
    print(data.x_train.shape, data.y_train.shape)
    print(data.x_test.shape, data.y_test.shape)