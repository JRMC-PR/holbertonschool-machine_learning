#!/usr/bin/env python3
"""This module contains the backward method for the BidirectionalCell class"""
import numpy as np


class BidirectionalCell:
    """This class represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """
        # initialize weight and baiaes values

        # Initialize the weight matrix for the forward hidden state.
        # The shape is (i + h, h), where:
        # i is the dimensionality of the input data,
        # h is the dimensionality of the hidden state.
        # This matrix is used to compute the next hidden state
        # in the forward direction.
        self.Whf = np.random.normal(size=(i + h, h))

        # Initialize the bias vector for the forward hidden state.
        # The shape is (1, h), where h is the dimensionality of
        # the hidden state.
        # This bias is added to the computation of the next hidden
        # state in the forward direction.
        self.bhf = np.zeros((1, h))

        # Initialize the weight matrix for the backward hidden state.
        # The shape is (i + h, h), where:
        # i is the dimensionality of the input data,
        # h is the dimensionality of the hidden state.
        # This matrix is used to compute the next hidden state in
        # the backward direction.
        self.Whb = np.random.normal(size=(i + h, h))

        # Initialize the bias vector for the backward hidden state.
        # The shape is (1, h), where h is the dimensionality of the
        # hidden state.
        # This bias is added to the computation of the next hidden state
        # in the backward direction.
        self.bhb = np.zeros((1, h))

        # Initialize the weight matrix for the output layer.
        # The shape is (2 * h, o), where:
        # h is the dimensionality of the hidden state,
        # o is the dimensionality of the output.
        # The factor of 2 accounts for the concatenation of the forward
        # and backward hidden states.
        self.Wy = np.random.normal(size=(2 * h, o))

        # Initialize the bias vector for the output layer.
        # The shape is (1, o), where o is the dimensionality of the output.
        # This bias is added to the computation of the output.
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """This method calculates the forward propagation for one time step
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
                    m: is the batch size
                    h: is the dimensionality of the hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data input
                 for the cell
                 m: is the batch size
                 i: is the dimensionality of the data
        Returns: h_next
                 h_next: the next hidden state
        """
        # Concatenate the previous hidden state and the input data
        # This combined input will be used to compute the forward hidden state
        # and the backward hidden state
        # axis=1 to concatenate horizontally
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        # Print the shape of the cell input
        # print(f"cell input shape: {cell_input.shape}")

        # Calculate the forward hidden state
        # The shape of the forward hidden state is (m, h)
        h_next = np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)
        # Print the shape of the forward hidden state
        # print(f"forward hidden state shape: {h_next.shape}")

        return h_next

    def backward(self, h_next, x_t):
        """This function caclulates  the hidden state in the
        backward direction for one time step
        Args:
            h_next: numpy.ndarray of shape (m, h) containing the
            next hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the
            data input for the cell
                m: is the batch size for the data
                i: is the dimensionality of the data
        Returns: h_pev
                h_prev: the previous hidden state
        """
        # Concatenate the next hidden state and the input data
        # This combined input will be used to compute the forward hidden state
        # and the backward hidden state
        # axis=1 to concatenate horizontally
        cell_input = np.concatenate((h_next, x_t), axis=1)
        # Print the shape of the cell input
        # print(f"cell input shape: {cell_input.shape}")

        # Calculate the backward hidden state
        # The shape of the backward hidden state is (m, h)
        h_prev = np.tanh(np.matmul(cell_input, self.Whb) + self.bhb)
        # Print the shape of the backward hidden state
        # print(f"backward hidden state shape: {h_prev.shape}")

        return h_prev

    def output(self, H):
        """This function callculates all the outputs for the RNN
        Args:
            H: numpy .ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding
            their initialized states
                t: is the number of time steps
                m: is the batch size for the data
                h: is the dimensionality of the hidden state
        Returns: Y
                Y: numpy.ndarray of shape (t, m, o) that contains the
                outputs
        """
        # since H contains the concatenated hidden states from both directions
        # we need to calculate the output for each direction
        # and concatenate them to get the final output
        # The shape of the output is (t, m, o)

        # Extract the shapes
        t, m, _ = H.shape

        # initialize the outputs array
        Y = np.zeros((t, m, self.Wy.shape[1]))

        # Calculate the output for each time step
        for i in range(t):
            # Calculate the output for the i-th time step.
            # H[i] is the concatenated hidden state for the i-th time
            # step, with shape (m, 2 * h).
            # Perform matrix multiplication between H[i] and the weight
            # matrix Wy, which has shape (2 * h, o).
            # Add the bias vector by, which has shape (1, o), to the result
            # of the matrix multiplication.
            # Apply the softmax function to the result to obtain the output
            # probabilities for the i-th time step.
            # Store the result in Y[i], which will have shape (m, o).
            Y[i] = self.softmax(np.matmul(H[i], self.Wy) + self.by)

        return Y

    def softmax(self, X):
        """This function calculates the softmax activation function
        for a given numpy.ndarray
        Args:
            X: numpy.ndarray of shape (m, n) that contains the data
            m: is the number of data points
            n: is the number of features in X
        Returns: Y
            Y: numpy.ndarray of shape (m, n) containing the softmax
            activation function
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
