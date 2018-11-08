from time import time
from typing import Callable, Optional, Union, Tuple

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.optimizers import RMSprop

import wandb
from wandb.keras import WandbCallback

from datasets.base import Dataset
from models.base import Model
from training.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = False
GPU_UTIL_SAMPLER = False

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_precisions = []
        self.val_profits = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()[:,0]
        val_targ = self.validation_data[1][:,0]
        
        true_positives = np.sum(np.round(val_predict * val_targ))
        false_positives = np.sum(np.round(val_predict * (1-val_targ)))
        predicted_positives = np.sum(np.round(val_predict))
        _val_precision = true_positives / (predicted_positives + 1e-7)
        _val_profit = 96 * true_positives - 104 * false_positives
        self.val_precisions.append(_val_precision)
        self.val_profits.append(_val_profit)
        print (f'- val_precision: {_val_precision} - val_profit: {_val_profit}')
        wandb.log({'val_precision': _val_precision, 'val_profit': _val_profit})
        return
        


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, learning_rate: float, gpu_ind: Optional[int]=None, use_wandb=False) -> Model:
    callbacks = []

    callbacks.append(Metrics())
    
    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilization = GPUUtilizationSampler(gpu_ind)
        callbacks.append(gpu_utilization)

    if use_wandb:
        wandb = WandbCallback()
        callbacks.append(wandb)

    model.network.summary()

    t = time()
    history = model.fit(dataset, batch_size, epochs, learning_rate, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return model
