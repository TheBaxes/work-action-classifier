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

    net = tf.layers.conv2d(inputs, 6, (3,3), strides=(1, 1), padding='same')
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.conv2d(net, 16, (3,3), strides=(1, 1), padding='same')
    net = tf.layers.max_pooling2d(net, (2,2), (2,2))
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 120, activation=tf.nn.relu)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    return tf.layers.dense(net, num_classes)


def model_fn(features, labels, mode, params):
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
    # Define a boolean to be True if running on TRAIN mode and False otherwise
    training = mode == tf.estimator.ModeKeys.TRAIN

    # Get the logits from the model given the image
    logits = model(features['image'], params['num_classes'], training)

    predictions = {
      # Compute the predictions by taking argmax of the logits
      "classes": tf.argmax(logits, axis=1),
      # Create probability predictions for each class by applying tf.nn.softmax
      # to de logits.
      "probabilities": tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes) usign 
    # tf.losses.sparse_softmax_cross_entropy
    xentropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = xentropy
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define the optimizer (e.g. tf.train.AdamOptimizer)
            optimizer = tf.train.AdamOptimizer(params['learning_rate'])

            # Define the training operation and include global_step using
            # tf.train.get_global_step()
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)