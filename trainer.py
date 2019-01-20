"""Trainer module

This module is responsible for training and evaluating the model. It is meant
to be run as a script as follows:
* `$ python trainer.py config.yml`: to train the model and compute performance
    metrics on training and validation data.
* `$ python trainer.py config.yml -e`: to evaluate the model on test data.

"""

import argparse
import os
import shutil

import tensorflow as tf
import yaml

import data
import model


def train_model(params):
    """
    Train the model given some hyperparameters of the model or the
    training procedure, it depends on calling the input_fn and model_fn
    defined in data.py and module.py respectively. This function
    does not return anything, but may save the trained model to disk.

    Args:
        params (dict): a dictionary with some settings and
            hyperparameters to use during training.

    """
    raise NotImplementedError


def eval_model(params):
    """
    This function evaluates the model with the test data

    Args:
        params (dict): a dictionary with some settings and
            hyperparameters to use during training.

    """
    raise NotImplementedError

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to configuration file")
    parser.add_argument('--warm-start', action='store_true',
        help="whether to start the model from checkpoints"
    )
    parser.add_argument('-v', '--verbosity', default='INFO',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'],
    )
    args = parser.parse_args()
    tf.logging.set_verbosity(args.verbosity)

    with open(args.config, 'r') as stream:
        params = yaml.load(stream)

    if not args.warm_start:
        try:
            shutil.rmtree(params['model_dir'])
        except FileNotFoundError:
            pass
        finally:
            os.makedirs(params['model_dir'])

    tf.logging.info("Using parameters: {}".format(params))

    train_model(params)