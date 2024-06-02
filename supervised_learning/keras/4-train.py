#!/usr/bin/env python3
"""Thisa module trains a model using
mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Train a model using mini-batch gradient descent.

    Parameters:
    network (keras model): The model to train.
    data (numpy.ndarray): The input data, of shape (m, nx).
    labels (numpy.ndarray): The labels of the data,
    one-hot encoded, of shape (m, classes).
    batch_size (int): The size of the batch used for
    mini-batch gradient descent.
    epochs (int): The number of passes through data for
    mini-batch gradient descent.
    verbose (bool, optional): Determines if output should
    be printed during training. Default is True.
    shuffle (bool, optional): Determines whether to shuffle
    the batches every epoch. Default is False.

    Returns:
    history: The History object generated after training the model.
    """

    # Train the model using mini-batch gradient descent.
    # The History object is a record of training loss values
    # and metrics values at successive epochs, as well as validation
    # loss values and validation metrics values (if applicable).
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
    )

    return history
