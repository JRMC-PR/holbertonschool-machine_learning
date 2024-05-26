#!/usr/bin/env python3
"""This module contains a function that conducts
forward propagation using Dropout.
"""
import numpy as np


def tanh_activation(Z):
    """
    Applies the tanh activation function.
    """
    return np.tanh(Z)


def softmax_activation(Z):
    """
    Applies the softmax activation function.
    """
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    X: numpy.ndarray - the input data for the network.
    weights: dict - the weights and biases of the neural network.
    L: int - the number of layers in the network.
    keep_prob: float - the probability that a node will be kept.

    Returns:
    A dictionary containing the outputs of each layer and
    the dropout mask used on each layer.
    """
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        # Calculate the pre-activation linear
        # combination of weights and inputs
        Z = (
            np.matmul(weights["W" + str(i)], cache["A" + str(i - 1)])
            + weights["b" + str(i)]
        )
        if i != L:
            # Apply tanh activation function and dropout
            A = tanh_activation(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A /= keep_prob
            cache["D" + str(i)] = D
        else:
            # Apply softmax activation function
            A = softmax_activation(Z)
        cache["A" + str(i)] = A

    return cache
