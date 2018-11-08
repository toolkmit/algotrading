import pathlib
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, Activation, LeakyReLU, BatchNormalization
from keras.models import Sequential
from keras.models import Model as KerasModel


def cnn(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        filters: int=32,
        kernel_size: int=6,
        fc_layer_size: int=32,
        dropout_pct: float=0.75,
        batch_norm_on: bool=False) -> KerasModel:
    
    num_classes = output_shape[0]
    
    model = Sequential()
    model.add(Conv1D(input_shape=input_shape,
                     filters=filters,
                     kernel_size=kernel_size,
                     padding='same',
                     activation='relu'))
    if batch_norm_on:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_pct))

    model.add(Conv1D(filters=filters*2,
                     kernel_size=kernel_size*2,
                     padding='same',
                     activation='relu'))
    if batch_norm_on:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_pct))

    model.add(Flatten())

    model.add(Dense(fc_layer_size, activation='relu'))
    if batch_norm_on:
        model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))
    
    return model