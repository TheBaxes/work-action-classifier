"""Model Module.

This module implements the specific mapping from input to output, as well as
the model_fn as required for TensorFlow's Estimator API. Everything that is 
related to the model architecture, the loss function and evaluation metrics 
is to be defined in this module.

Different models may use different arguments so it is ok if the definition
of ``model`` changes over versions.

Todo:
    * Implement everything.

"""

import tensorflow as tf


def model(inputs, num_classes, training):
    """
    This function implements the model architecture. This function is
    rensponsible for mapping a batch of RGB images to the output of
    the last leyer of the neural network.
    
    Args:
        inputs (tf.Tensor): a batch of RGB images in NHWC format.
        num_classes (int): number of classes of the classification problem,
            which corresponds to the output of the last layer of the
            neural network.
        training (bool): whether the model should be run in trainng or 
            inference mode. This is particularly important when using modules
            like dropout or batch normalization.
        
    Returns:
        (tf.Tensor): The output of the last layer of the model. It must have
        shape (N, num_classes).

    """

    raise NotImplementedError


def resnet_fn(features, labels, mode, params):
    """Model function required by the tf.estimator API

    This function is ment to be called with proper arguments by TensorFlow
    through its Estimator API.

    Args:
        features: Dictionary with the with the 'image' keyword, which
            corresponds to a batch of RGB images in NHWC format.
        labels: Tensor with image clases labels.
        mode: the mode in which the function is called. Should be one
            of 'eval', 'train', 'predict'. This parameter is passed
            by tensorflow automatically.
        params (dict): a dictionary with the hyperparameters of the
            model or the training process.

    Returns:
        An instance of tf.estimator.EstimatorSpec

    """  

    raise NotImplementedError
