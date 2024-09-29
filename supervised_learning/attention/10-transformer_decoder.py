#!/usr/bin/env python3
"""This module contains the Decoder class
    that inherits from tensorflow.keras.layers.Layer"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('9-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """This class creates a decoder for a transformer"""
    def __init__(
            self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """Class constructor
            Args:
                N (int): the number of blocks in the encoder
                dm (int): the dimensionality of the model
                h (int): the number of heads
                hidden (int): the number of hidden units in the fully
                connected layer
                target_vocab (int): the size of the target vocabulary
                max_seq_len (int): the maximum sequence length possible
                drop_rate (float): the dropout rate
        """
        # set the number of blocks
        self.N = N
        # Set the dementionality of the model
        self.dm = dm
        # Set the embedding layer
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        # Set the positional encoding layer
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # Set the list of decoder blocks
        self.blocks = [DecoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        # Set the dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """This method calls the decoder
            Args:
                x (tf.Tensor): contains the input to the decoder
                encoder_output (tf.Tensor): contains the output of the encoder
                training (bool): determines if the model is in training
                look_ahead_mask (tf.Tensor): contains the mask to be applied to
                the first multi head attention layer
                padding_mask (tf.Tensor): contains the mask to be applied to
                the second multi head attention layer
            Returns:
                (tf.Tensor): contains the decoder's output
        """
        seq_len = x.shape[1]

        # apply the embedding layer
        x = self.embedding(x)
        # scale the embedding by the square root of the dimension
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # add the positional encoding
        x += self.positional_encoding[:seq_len]

        # apply the dropout layer
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
