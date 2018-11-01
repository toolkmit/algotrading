from time import time
from typing import Callable, Optional, Union, Tuple

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop

import wandb
from wandb.keras import WandbCallback

from datasets.base import Dataset
from models.base import Model
from training.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = False
GPU_UTIL_SAMPLER = False


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, gpu_ind: Optional[int]=None, use_wandb=False) -> Model:
    callbacks = []

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
    history = model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return model
