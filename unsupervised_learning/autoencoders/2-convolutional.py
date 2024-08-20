#!/usr/bin/env python3
"""This module contains a convolutional autoencoder function"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """This function creates a convolutional autoencoder
    Each convolution in the encoder should use a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation, followed
    by upsampling of size (2, 2)
    The second to last convolution should instead use valid padding
    The last convolution should have the same number of filters as the number
    of channels in input_dims with sigmoid activation and no upsampling
    Args:
        input_dims: integer containing the dimensions of the model input
        filters: list containing the number of filters for each convolutional
                 layer in the encoder
        latent_dims: integer containing the dimensions of the latent space
                     representation
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # Step 1: Define the encoder model
    # create the input layer for the encoder
    encoder_inputs = keras.Input(shape=input_dims)
    # print for data visualization
    # print(f"---" * 20)
    # print(f"input_dims: {input_dims}\n\nencodes_inputs: {encoder_inputs}")
    # print(f"---" * 20)

    # create the encoder layers
    for idx, units in enumerate(filters):
        # Add convolutional layers with the relu activation function
        layer = keras.layers.Conv2D(
            filters=units,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
        # print(f"---" * 20)
        # print(f"layer: {layer}")
        # print(f"---" * 20)
        if idx == 0:
            # if it is the first layer, set the input
            outputs = layer(encoder_inputs)
            # print(f"---" * 20)
            # print(f"outputs: {outputs.get_config()}")
            # print(f"---" * 20)
        else:
            # if it is not the first layer, set the output of the
            # previous layer
            outputs = layer(outputs)
        # print(f"---" * 20)
        # print(f"output: {outputs.get_config()}")
        # print(f"---" * 20)
        # Add max pooling layers
        layer = keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=None, padding="same"
        )
        # print(f"---" * 20)
        # print(f"layer: {layer.get_config()}")
        # print(f"---" * 20)

        # make the max pooling layer the output layer for the encoder
        outputs = layer(outputs)
        # print(f"---" * 20)
        # print(f"outputs after max pooling: {outputs.get_config()}")
        # print(f"---" * 20)
    # create the encoder model
    encoder = keras.models.Model(inputs=encoder_inputs, outputs=outputs)
    # print(f"---" * 20)
    # print(f"encoder summary: {encoder.summary()}")
    # print(f"---" * 20)

    # Step 2: Define the decoder model
    decoder_inputs = keras.Input(shape=latent_dims)
    # create the decoder layers
    # iterate over the filters in reverse order
    for idx, units in enumerate(reversed(filters)):
        # print(f"---" * 20)
        # print(f"idx: {idx}\n\nunits: {units}")
        # print(f"---" * 20)
        if idx != len(filters) - 1:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            )
            # print(f"---" * 20)
            # print(f"layer: {layer.get_config()}")
            # print(f"---" * 20)
            if idx == 0:
                outputs = layer(decoder_inputs)
                # print(f"---" * 20)
                # print(f"outputs: {outputs.get_config()}")
                # print(f"---" * 20)
            else:
                outputs = layer(outputs)
                # print(f"---" * 20)
                # print(f"outputs: {outputs.get_config()}")
                # print(f"---" * 20)
        else:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )
            # print(f"---" * 20)
            # print(f"layer: {layer.get_config()}")
            # print(f"---" * 20)
            outputs = layer(outputs)
            # print(f"---" * 20)
            # print(f"outputs: {outputs.get_config()}")
            # print(f"---" * 20)

        layer = keras.layers.UpSampling2D(size=(2, 2))
        # print(f"---" * 20)
        # print(f"layer: {layer.get_config()}")
        # print(f"---" * 20)
        outputs = layer(outputs)
        # print(f"---" * 20)
        # print(f"outputs: {outputs.get_config()}")
        # print(f"---" * 20)
    layer = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )
    # print(f"---" * 20)
    # print(f"layer: {layer.get_config()}")
    # print(f"---" * 20)
    outputs = layer(outputs)
    # print(f"---" * 20)
    # print(f"outputs: {outputs.get_config()}")
    # print(f"---" * 20)
    decoder = keras.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the full autoencoder model
    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs)
    auto = keras.models.Model(inputs=encoder_inputs, outputs=outputs)

    # print for data visualization
    # print(f"---" * 20)
    # print(f"encoder: {encoder}")
    # print(f"---" * 20)
    # print(f"decoder: {decoder}")
    # print(f"---" * 20)
    # print(f"auto: {auto.summary()}")
    # print(f"---" * 20)

    # compile the autoencoder
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
