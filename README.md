# TensorFlow's Estimator and Data APIs

This project is a hands on tutorial about TensorFlow's Estimator and Data APIs, which by the the time of the version 1.12 is the preferred way to make machine learning models with TensorFlow. These API's allow you to easily and efficiently build and train custom models and also guide you to make your code clean and well structured. I should mention that other people may use these APIs differently, so I will just show the structure I think is best, for that with devide the code in three main modules:

* `data.py`, which will take care of all neccessary steps for the data pipeline, that is: feed training data to train the model, feed validation data to evaluate the model, feed the model with input data for predictions. Note that these steps may require take the data from any source neccessary, but the source depends on the particular project.
* `model.py`, in which we define the particular machine learning model (i.e. how we pass from input to output) as well as the loss function and optimization algorithm.
* `trainer.py`, which takes the previous two modules and is responsible for training, evaluation and even deployment (although we will not cover deployment here, the Estimator API will automatically save the trained model).

Moreover, we will have a separte `config.yml` file for keepin training and other configuration parameters.

The `data.py` module can be run as an script to prepare the dataset, which means downloading it and putting it into training, validation and testing subsets of the data, following thid format:

```
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
```
Note that TensorFlow has a preferred way of storing training data: [TensorFlow Records](https://www.tensorflow.org/tutorials/load_data/tf-records). They are particulary useful for performance reasons, but I have decided to omit them from this guide.

`trainer.py` is meant to be run as a script as follows:
* `$ python trainer.py config.yml`: to train the model and compute performance metrics on training and validation data.
* `$ python trainer.py config.yml -e`: to evaluate the model on test data.


## Repo structure
This repo is meant to be used for a tutorial, so its branches are organized to represent different stages of the tutorial:
* `master` shows the template of the project defines functionality but does not actually show any implementation.
* `partial` implements part of the functionality and guives some guides to implement the rest.
* `full` implements all defined functionality.
