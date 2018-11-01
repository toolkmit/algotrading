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
        window_width: float=16,
        window_stride: float=8) -> KerasModel:
    
    num_classes = output_shape[0]
    
    model = Sequential()
    model.add(Conv1D(input_shape=input_shape,
                     filters=32,
                     kernel_size=2,
                     padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))

    model.add(Conv1D(filters=64,
                     kernel_size=4,
                     padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))

    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))
    
    return model