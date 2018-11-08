from typing import Callable, Dict
import pathlib

import numpy as np
from keras.models import Model as KerasModel
from keras.optimizers import Adam

from datasets.base import Dataset


DIRNAME = pathlib.Path(__file__).parents[1].resolve() / 'weights'


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict=None, network_args: Dict=None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()


    @property
    def weights_filename(self):
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

    def fit(self, dataset, batch_size=64, epochs=10, learning_rate=1e-3, callbacks=[]):
        self.network.compile(loss=self.loss(), optimizer=self.optimizer(lr=learning_rate), metrics=self.metrics())

        self.network.fit(
            dataset.x_train,
            dataset.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=callbacks,
            validation_data=(dataset.x_test, dataset.y_test),
            shuffle=True
        )

    def evaluate(self, x, y):
        preds = self.network.predict(x, batch_size=16)   # Use a small batch size to use less memory
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):
        return 'categorical_crossentropy'

    def optimizer(self, lr):
        return Adam(lr=lr)

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)