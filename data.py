"""Data module

This module defines functions and operations for handling all stages of the 
data processing pipeline. Moreover, it may be used as a command line application
to prepare the dataset in the following format:

dataset/
    training/
        lbl0/
            img1.jpg (or other format)
            img2.jpg
        lbl1/
            ...
        ...
    validation/
        ...
    testing/
        ...

To do so, you can run any of the following commands:

    $ python data.py --url=URL_WITH_COMPRESSED_DATA
    $ python data.py --fpath=FILEPATH_OF_COMPRESSED_DOWNLOAD

The second example is to be used only if data has already been downloaded. This
module holds defaults for the url and dataset file paths and name dirs as
attributes.

Attributes:
    DEFAULT_URL (str): default url where to get the compressed data.
    DEFAULT_COMPRESSED_FPATH (str): default path for the compressed data after
        download.

Todo:
    * Implement everything.
"""

import os

import tensorflow as tf

DEFAULT_URL = 'benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
DEFAULT_COMPRESSED_FPATH = 'dataset.zip'

def download_compressed_dataset(url):
    """Downloads the dataset in compressed format.
    
    This function does not return anything
    """
    raise NotImplementedError

def prepare_dataset(fpath):
    """Prepare the dataset form the download file.
    
    This function takes the filepath of the download file and prepares the data
    by dividing into 3 folders: training, validation and testing, each of which
    has examples of every available label in a proportion roughly to 70%, 20% and
    10% respectively

    This function does not return anything

    """
    raise NotImplementedError


def find_sources(data_dir, mode='training', shuffle=True):
    """List all sources of data with the respective label. 

    Args:
        data_dir (str): path to the directory with the data.
        mode (str): whether using 'training', 'validation' or 'testing'
            data. Defaults to 'training'
        shuffle (bool): whether to shuffle the list before output.

    Returns:
        a list of tuples of (path/to/imagefile.ppm, label), where lable is an
        integer in [0, num_classes). This function should usually be used
        to return the sources for the input_fn function defined bellow.

    """
    raise NotImplementedError

def input_fn(sources, train, params):
    """Input function required by the estimator API.

    Args:
        sources (list): a list of tuples of (path/to/imagefile.ppm, label),
            where lable is an integer in [0, num_classes).
        train (bool): whether the input_fn will serve data for training (True)
            or evaluation purposes (False). Data may have different treatment
            depending on this value (e.g. training data should be shuffled).
        params (dict): dictionary holding parameters for some hyperparameters
            when feeding the data (e.g. batch_size and image_size).

    Returns:
        A tuple of (features, label). Features is a dictionary holding all
        relevant features by key (e.g. image tensor) and label is a tensor
        of integers.

    """
    
    images, labels = zip(*sources)
    images, labels = list(images), list(labels)

    images = tf.data.Dataset.from_tensor_slices(images)
    # Read images as binary data
    images = images.map(lambda img: tf.read_file(img))
    # Decode binary images
    images = images.map(lambda img: tf.image.decode_image(img, channels=3))
    # Cast the images to float32
    images = 
    # Normalize the images by dividing the images by 255.0
    images = 
    # Resize the images to the expected size given by the network
    images = 

    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels))

    if train:
        ds = ds.shuffle(buffer_size=params['shuffle_buffer'])
        ds = ds.repeat(params['num_epochs'])

    ds = ds.batch(params['batch_size'])
    iterator = ds.make_one_shot_iterator()
    images, labels = iterator.get_next()

    features = {'image': images} 
    
return features, labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', 
        help='download link for the original training')
    parser.add_argument('--fpath',
        help="Path to the compressed download for the original tarining data")
    
    
    args = parser.parse_known_args()

    if not os.path.exists(args.fpath):
        download_compressed_dataset(args.url)
    prepare_dataset(args.fpath)

