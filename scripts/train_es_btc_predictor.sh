#!/bin/sh
python training/run_experiment.py '{"dataset": "ESBTCDataset", "model": "ESModel", "network": "cnn", "train_args": {"learning_rate": 1e-4}}'