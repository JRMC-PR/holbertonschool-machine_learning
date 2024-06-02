#!/usr/bin/env pyton3
"""This moduel set up Adam optimization for a keras model
with categorical crossentropy loss and accuracy metrics
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics

    Args:
        network (keras.model): model to optimize
        alpha (float): learning rate
        beta1 (float): first Adam optimization parameter
        beta2 (float): second Adam optimization parameter

    Returns:
        None
    """
    # Define the optimizer. Adam is an optimization algorithm that
    # can be used instead
    # of the classical stochastic gradient descent procedure to
    #  network weights
    # iteratively based on the training data.
    optimizer = \
        K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)

    # Compile the model with the optimizer, the categorical crossentropy loss
    # function, and the accuracy metric.
    # Categorical crossentropy is a loss function that is used
    # in multi-class
    # classification tasks. It is the loss function to be evaluated
    # first and minimized.
    # The accuracy metric computes the mean accuracy rate across all
    # predictions
    # for multi-class classification problems.
    network.compile(
        optimizer=optimizer, loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
