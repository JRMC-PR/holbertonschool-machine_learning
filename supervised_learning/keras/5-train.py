#!/usr/bin/env python3
"""This module trains a model using mini-batch gradient descent
also analyzes validaiton data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Train a model using mini-batch gradient descent and
    also analyze validation data.

    Parameters:
    network (keras model): The model to train.
    data (numpy.ndarray): The input data, of shape (m, nx).
    labels (numpy.ndarray): The labels of the data,
    one-hot encoded, of shape (m, classes).
    batch_size (int): The size of the batch used for
    mini-batch gradient descent.
    epochs (int): The number of passes through data for
    mini-batch gradient descent.
    validation_data (tuple): Data to validate the model with, if not None.
    verbose (bool, optional): Determines if output should
    be printed during training. Default is True.
    shuffle (bool, optional): Determines whether to shuffle
    the batches every epoch. Default is False.

    Returns:
    history: The History object generated after training the model.
    """

    # Fit the model using the provided parameters. If validation_data is
    # not None, it will be used for validation.
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )

    return history
