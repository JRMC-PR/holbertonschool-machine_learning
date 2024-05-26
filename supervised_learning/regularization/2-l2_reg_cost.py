#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2
    regularization.

    Parameters:
    cost: tensor - a tensor containing the cost of the network
    without L2 regularization.
    model: Keras model - a Keras model that includes layers with
    L2 regularization.

    Returns:
    A tensor containing the total cost for each layer of the
    network, accounting for L2 regularization.
    """
    # Get the list of regularization losses from the model
    reg_losses = model.losses

    # Add the regularization losses to the original cost
    total_cost = cost + tf.add_n(reg_losses)

    return total_cost
