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
import shutil
import zipfile
from pathlib import Path
import random

import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

DEFAULT_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
DEFAULT_COMPRESSED_FPATH = 'dataset.zip'

def download_compressed_dataset(url=DEFAULT_URL, filename=DEFAULT_COMPRESSED_FPATH, chunk_size=2000):
    """Downloads the dataset in compressed format.

    This function does not return anything
    """
    nfile = requests.get(url, stream=True)

    with open(str(Path(filename)), 'wb') as f:
        for chunk in nfile.iter_content(chunk_size):
            f.write(chunk)

def prepare_dataset(fpath=DEFAULT_COMPRESSED_FPATH):
    """Prepare the dataset form the download file.

    This function takes the filepath of the download file and prepares the data
    by dividing into 3 folders: training, validation and testing, each of which
    has examples of every available label in a proportion roughly to 70%, 20% and
    10% respectively

    This function does not return anything

    """

    dataset_folder = 'raw_dataset'
    test_fraction = 0.2
    train_fraction = 0.7

    if not os.path.exists(dataset_folder):
        with zipfile.ZipFile(fpath, "r") as z:
            z.extractall(dataset_folder)
    

    data_train = []
    data_test = []

    for folder in os.listdir(os.path.join(dataset_folder, "GTSRB", "Final_Training", "Images")):

        folder_data = glob(os.path.join(dataset_folder, "GTSRB", "Final_Training", "Images", folder, "*.ppm"))


        folder_data = np.array(folder_data)

        folder_data = np.random.permutation(folder_data)

        N = int(len(folder_data) * train_fraction)
        data_train.append(folder_data[:N])
        data_test.append(folder_data[N:])

    data_train = np.hstack(data_train)
    data_test = np.hstack(data_test)
    print(data_train[:10])
    print(len(data_train))
    print(len(data_test))
    
    for dataset_name, dataset in zip(["train", "test"], [data_train, data_test]):
        

        for filepath in tqdm(dataset, total=len(dataset)):
            folder, name = filepath.split(os.sep)[-2:]
            name = name.replace(".ppm", ".jpg")

            new_image_path = os.path.join("dataset", dataset_name, folder, name)
            os.makedirs(os.path.join("dataset", dataset_name, folder), exist_ok=True)

            with Image.open(filepath) as img:
                img.save(new_image_path)
                

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
    dir_name = 'train' if mode == 'training' else 'test'
    dir_name = os.path.join(data_dir, dir_name)
    label_to_files = {
        label: os.listdir(os.path.join(dir_name, label))
        for label in os.listdir(dir_name)
    }

    sources = []
    for label, fnames in label_to_files.items():
        for fname in fnames:
            fpath = os.path.join(dir_name, label, fname)
            sources.append((fpath, int(label)))

    if shuffle:
        random.shuffle(sources)

    return sources
    

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
    images = images.map(lambda img: tf.image.decode_jpeg(img, channels=3))
    # Cast the images to float32
    images = images.map(lambda img: tf.to_float(img))
    # Normalize the images by dividing the images by 255.0
    images = images.map(lambda img: img/255.0)
    # Resize the images to the expected size given by the network
    def helper(img):
        hh = params['image_size']
        img = tf.image.resize_images(img, (hh, hh))
        return img
    images = images.map(helper)

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
    parser.add_argument('--url', default=DEFAULT_URL,
        help='download link for the original training')
    parser.add_argument('--fpath',  default=DEFAULT_COMPRESSED_FPATH,
        help="Path to the compressed download for the original training data")

    args = parser.parse_args()

    if not os.path.exists(args.fpath):
        download_compressed_dataset(args.url)
    prepare_dataset(args.fpath)