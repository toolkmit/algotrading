import pathlib
from typing import Callable, Dict, Tuple

import numpy as np
import tensorflow
from keras import backend as K
from keras.models import load_model

from models.base import Model
from datasets.es_btc import ESBTCDataset
from networks.cnn import cnn


def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
        
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

class ESModel(Model):
    def __init__(self, dataset_cls: type=ESBTCDataset, network_fn: Callable=cnn, dataset_args: Dict=None, network_args: Dict=None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
        
    def metrics(self):
        return ['accuracy']