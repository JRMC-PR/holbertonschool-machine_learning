#!/usr/bin/env python3
"""This modlue contains the MultiHeadAttention class"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """This class performs multi head attention"""
    def __init__(self, dm, h):
        """Class constructor for MultiHeadAttention
            Args:
                dm (int): the dimensionality of the model
                h (int): the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """This method calls the MultiHeadAttention layer
            Args:
                Q (tf.Tensor): contains the query matrix
                K (tf.Tensor): contains the key matrix
                V (tf.Tensor): contains the value matrix
                mask (tf.Tensor): contains the optional mask
            Returns:
                tf.Tensor, tf.Tensor: contains the output, and the weights
        """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        def split_heads(x, batch_size):
            """This method splits the last dimension of tensor x into
            (h, depth)
                Args:
                    x (tf.Tensor): the tensor to split
                    batch_size (int): the batch size
                Returns:
                    tf.Tensor: the split tensor
            """
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        q = split_heads(q, batch_size)
        k = split_heads(k, batch_size)
        v = split_heads(v, batch_size)

        scaled_attention, weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (
            batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, weights
