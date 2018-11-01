#!/usr/bin/env python
import argparse
import json
import importlib
from typing import Dict
import os

import wandb

from training.util import train_model


DEFAULT_TRAIN_ARGS = {
    'batch_size': 64,
    'epochs': 100
}


def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool=True):
    """
    experiment_config is of the form
    {
        "dataset": "EmnistLinesDataset",
        "dataset_args": {
            "max_overlap": 0.4
        },
        "model": "LineModel",
        "network": "line_cnn_sliding_window",
        "network_args": {
            "window_width": 14,
            "window_stride": 7
        },
        "train_args": {
            "batch_size": 128,
            "epochs": 10
        }
    }
    save_weights: if True, will save the final model weights to a canonical location (see Model in models/base.py)
    gpu_ind: integer specifying which gpu to use
    """
    print(f'Running experiment with config {experiment_config} on GPU {gpu_ind}')

    datasets_module = importlib.import_module('datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)
    
    models_module = importlib.import_module('models')
    model_class_ = getattr(models_module, experiment_config['model'])
    
    networks_module = importlib.import_module('networks')
    network_fn_ = getattr(networks_module, experiment_config['network'])
    network_args = experiment_config.get('network_args', {})
    model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn_, dataset_args=dataset_args, network_args=network_args)
    print(model)
    
    experiment_config['train_args'] = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)
    experiment_config['gpu_ind'] = gpu_ind
    
    if use_wandb:
        wandb.init()
        wandb.config.update(experiment_config)
    
    train_model(
        model,
        dataset,
        epochs=experiment_config['train_args']['epochs'],
        batch_size=experiment_config['train_args']['batch_size'],
        gpu_ind=gpu_ind,
        use_wandb=use_wandb
    )
    score = model.evaluate(dataset.x_test, dataset.y_test)
    print(f'Test evaluation: {score}')

    if use_wandb:
        wandb.log({'test_metric': score})

    if save_weights:
        model.save_weights()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use."
    )
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="JSON of experiment to run (e.g. '{\"dataset\": \"EmnistDataset\", \"model\": \"CharacterModel\", \"network\": \"mlp\"}'"
    )
    args = parser.parse_args()


    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    run_experiment(experiment_config, args.save, args.gpu)