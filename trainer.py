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
    estimator = tf.estimator.Estimator(
    model.model_fn,
    model_dir=params['model_dir'],
    params=params,
    config=tf.estimator.RunConfig(
            save_checkpoints_steps=params['save_checkpoints_steps'],
            save_summary_steps=params['save_summary_steps'],
            log_step_count_steps=params['log_frequency'],
            keep_checkpoint_max=3
        )
    )

    sources = data.find_sources(params['data_dir'], mode='training')
    train_spec = tf.estimator.TrainSpec(
        lambda: data.input_fn(sources, True, params),
        max_steps=params['max_steps']
    )

    sources = data.find_sources(params['data_dir'], mode='testing')
    eval_spec = tf.estimator.EvalSpec(
        lambda: data.input_fn(sources, False, params),
        steps=params['eval_steps'],
        start_delay_secs=params['start_delay_secs'],
        throttle_secs=params['throttle_secs']
    )

    tf.logging.info("Start experiment....")

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def eval_model(params):
    """
    This function evaluates the model with the test data

    Args:
        params (dict): a dictionary with some settings and
            hyperparameters to use during training.

    """

    # Go through the documentation tf.estimator.Estimator (https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#evaluate)
    # and try to implement this function.
    raise NotImplementedError

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to configuration file")
    parser.add_argument('-w', '--warm-start', action='store_true',
        help="whether to start the model from checkpoints")
    parser.add_argument('-e', '--evaluate', action='store_true',
        help="whether to evaluate the model instead of training it")
    parser.add_argument('-v', '--verbosity', default='INFO',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'])

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

    if args.evaluate:
        eval_model(params)
    else:
        train_model(params)